package mlstate

import (
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	"io/ioutil"
	"os"
	py "pfi/sensorbee/py/p"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"sync"
)

var (
	lossPath = data.MustCompilePath("loss")
	accPath  = data.MustCompilePath("accuracy")

	ErrAlreadyTerminated = errors.New("PyMLState is already terminated")
)

// PyMLState is python instance specialized to multiple layer classification.
// The python instance and this struct must not be coppied directly by assignment
// statement because it doesn't increase reference count of instance.
type PyMLState struct {
	modulePath string
	moduleName string
	className  string

	ins *py.ObjectInstance

	bucket    data.Array
	batchSize int

	rwm sync.RWMutex
}

type pyMLMsgpack struct {
	ModulePath string `codec:"module_path"`
	ModuleName string `codec:"module_name"`
	ClassName  string `codec:"class_name"`
	BatchSize  int    `codec:"batch_size"`
}

// NewPyMLState creates `core.SharedState` for multiple layer classification.
func NewPyMLState(modulePathName, moduleName, className string, batchSize int,
	params data.Map) (*PyMLState, error) {
	ins, err := newPyInstance("create", modulePathName, moduleName, className, []data.Value{params}...)
	if err != nil {
		return nil, err
	}

	s := &PyMLState{}
	s.set(ins, modulePathName, moduleName, className, batchSize)
	return s, nil
}

// newPyInstance creates a new Python class instance.
// User must call DecRef method to release a resource.
func newPyInstance(createMethodName, modulePathName, moduleName, className string, args ...data.Value) (py.ObjectInstance, error) {
	var null py.ObjectInstance
	py.ImportSysAndAppendPath(modulePathName)

	mdl, err := py.LoadModule(moduleName)
	if err != nil {
		return null, err
	}
	defer mdl.DecRef()

	class, err := mdl.GetClass(className)
	if err != nil {
		return null, err
	}
	defer class.DecRef()

	ins, err := class.CallDirect(createMethodName, args...)
	return py.ObjectInstance{ins}, err
}

// set sets or overwrites fields of PyMLState struct.
// This method steals a reference of ins object.
// Caller must not call `s.ins.DecRef()` because set calls DecRef for old ins object.
func (s *PyMLState) set(ins py.ObjectInstance, modulePathName, moduleName, className string,
	batchSize int) {
	if s.ins != nil {
		s.ins.DecRef()
	}

	s.modulePath = modulePathName
	s.moduleName = moduleName
	s.className = className
	s.ins = &ins
	s.bucket = make(data.Array, 0, batchSize)
	s.batchSize = batchSize
}

// Terminate this state.
func (s *PyMLState) Terminate(ctx *core.Context) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if s.ins == nil {
		return nil // This isn't an error in Terminate.
	}
	s.ins.DecRef()
	return nil
}

// Write and call "fit" function. Tuples is cached per train batch size.
func (s *PyMLState) Write(ctx *core.Context, t *core.Tuple) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if s.ins == nil {
		return ErrAlreadyTerminated
	}
	s.bucket = append(s.bucket, t.Data)

	var err error
	if len(s.bucket) >= s.batchSize {
		m, er := s.fit(ctx, s.bucket)
		err = er
		s.bucket = s.bucket[:0] // clear slice but keep capacity

		// optional logging, return non-error even if the value does not have
		// accuracy and loss.
		if ret, er := data.AsMap(m); er == nil {
			var loss float64
			if l, e := ret.Get(lossPath); e != nil {
				return err
			} else if loss, e = data.ToFloat(l); e != nil {
				return err
			}
			var acc float64
			if a, e := ret.Get(accPath); e != nil {
				return err
			} else if acc, e = data.ToFloat(a); e != nil {
				return err
			}
			ctx.Log().Debugf("loss=%.3f acc=%.3f", loss/float64(s.batchSize),
				acc/float64(s.batchSize))
		}
	}

	return err
}

// Fit receives `data.Array` type but it assumes `[]data.Map` type
// for passing arguments to `fit` method.
func (s *PyMLState) Fit(ctx *core.Context, bucket data.Array) (data.Value, error) {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	if s.ins == nil {
		return nil, ErrAlreadyTerminated
	}
	return s.fit(ctx, bucket)
}

// fit is the internal implementation of Fit. fit doesn't acquire the lock nor
// check s.ins == nil. RLock is sufficient when calling this method because
// this method itself doesn't change any field of PyMLState. Although the model
// will be updated by the data, the model is protected by Python's GIL. So,
// this method doesn't require a write lock.
func (s *PyMLState) fit(ctx *core.Context, bucket data.Array) (data.Value, error) {
	return s.ins.Call("fit", bucket)
}

// FitMap receives `[]data.Map`, these maps are converted to `data.Array`
func (s *PyMLState) FitMap(ctx *core.Context, bucket []data.Map) (data.Value, error) {
	args := make(data.Array, len(bucket))
	for i, v := range bucket {
		args[i] = v
	}

	s.rwm.RLock() // Same as fit method. This doesn't have to be Lock().
	defer s.rwm.RUnlock()
	if s.ins == nil {
		return nil, ErrAlreadyTerminated
	}
	return s.ins.Call("fit", args)
}

// Predict applies the model to the data. It returns a result returned from
// Python script.
func (s *PyMLState) Predict(ctx *core.Context, dt data.Value) (data.Value, error) {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	if s.ins == nil {
		return nil, ErrAlreadyTerminated
	}
	return s.ins.Call("predict", dt)
}

// Save saves the model of the state. pystate calls `save` method and
// use its return value as dumped model.
func (s *PyMLState) Save(ctx *core.Context, w io.Writer, params data.Map) error {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	if s.ins == nil {
		return ErrAlreadyTerminated
	}

	if err := s.savePyMLMsgpack(w); err != nil {
		return err
	}

	temp, err := ioutil.TempFile("", "sensorbee_py_ml_state") // TODO: TempDir should be configurable
	if err != nil {
		return fmt.Errorf("cannot create a temporary file for saving data: %v", err)
	}
	filepath := temp.Name()
	if err := temp.Close(); err != nil {
		ctx.ErrLog(err).WithField("filepath", filepath).Warn("Cannot close the temporary file")
	}
	defer func() {
		if err := os.Remove(filepath); err != nil && !os.IsNotExist(err) {
			ctx.ErrLog(err).WithField("filepath", filepath).Warn("Cannot remove the temporary file")
		}
	}()

	_, err = s.ins.Call("save", data.String(filepath), params)
	if err != nil {
		return err
	}

	f, err := os.Open(filepath)
	if err != nil {
		return fmt.Errorf("cannot open the temporary file having the saved data: %v", err)
	}
	defer func() {
		if err := temp.Close(); err != nil {
			ctx.ErrLog(err).WithField("filepath", filepath).Warn("Cannot close the temporary file")
		}
	}()
	_, err = io.Copy(w, f)
	return err
}

const (
	pyMLStateFormatVersion uint8 = 1
)

func (s *PyMLState) savePyMLMsgpack(w io.Writer) error {
	if _, err := w.Write([]byte{pyMLStateFormatVersion}); err != nil {
		return err
	}

	// Save parameter of PyMLState before save python's model
	save := &pyMLMsgpack{
		ModulePath: s.modulePath,
		ModuleName: s.moduleName,
		ClassName:  s.className,
		BatchSize:  s.batchSize,
	}

	msgpackHandle := &codec.MsgpackHandle{}
	var out []byte
	enc := codec.NewEncoderBytes(&out, msgpackHandle)
	if err := enc.Encode(save); err != nil {
		return err
	}

	// Write size of pyMLMsgpack
	dataSize := uint32(len(out))
	err := binary.Write(w, binary.LittleEndian, dataSize)
	if err != nil {
		return err
	}

	// Write pyMLMsgpack in msgpack
	n, err := w.Write(out)
	if err != nil {
		return err
	}

	if n < len(out) {
		return errors.New("cannot save the pyMLMsgpack data")
	}

	return nil
}

// Load loads the model of the state. pystate calls `load` method and
// pass to the model data by using method parameter.
func (s *PyMLState) Load(ctx *core.Context, r io.Reader, params data.Map) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if s.ins == nil {
		return ErrAlreadyTerminated
	}

	var formatVersion uint8
	if err := binary.Read(r, binary.LittleEndian, &formatVersion); err != nil {
		return err
	}

	// TODO: remove PyMLState specific parameters from params

	switch formatVersion {
	case 1:
		return s.loadPyMsgpackAndDataV1(ctx, r, params)
	default:
		return fmt.Errorf("unsupported format version of PyMLState container: %v", formatVersion)
	}
}

func (s *PyMLState) loadPyMsgpackAndDataV1(ctx *core.Context, r io.Reader, params data.Map) error {
	var dataSize uint32
	if err := binary.Read(r, binary.LittleEndian, &dataSize); err != nil {
		return err
	}
	if dataSize == 0 {
		return errors.New("size of pyMLMsgpack must be greater than 0")
	}

	// Read pyMLMsgpack from reader
	buf := make([]byte, dataSize)
	n, err := r.Read(buf)
	if err != nil {
		return err
	}
	if n != int(dataSize) {
		return errors.New("read size is different from pyMLMsgpack")
	}

	// Desirialize pyMLMsgpack
	var saved pyMLMsgpack
	msgpackHandle := &codec.MsgpackHandle{}
	dec := codec.NewDecoderBytes(buf, msgpackHandle)
	if err := dec.Decode(&saved); err != nil {
		return err
	}

	temp, err := ioutil.TempFile("", "sensorbee_py_ml_state") // TODO: TempDir should be configurable
	if err != nil {
		return fmt.Errorf("cannot create a temporary file to store the data to be loaded: %v", err)
	}
	filepath := temp.Name()
	tempClosed := false
	closeTemp := func() {
		if tempClosed {
			return
		}
		if err := temp.Close(); err != nil {
			ctx.ErrLog(err).WithField("filepath", filepath).Warn("Cannot close the temporary file")
		}
		tempClosed = true
	}
	defer func() {
		closeTemp()
		if err := os.Remove(filepath); err != nil {
			ctx.ErrLog(err).WithField("filepath", filepath).Warn("Cannot remove the temporary file")
		}
	}()
	if _, err := io.Copy(temp, r); err != nil {
		return err
	}
	closeTemp()

	ins, err := newPyInstance("load", saved.ModulePath, saved.ModuleName, saved.ClassName, []data.Value{data.String(filepath), params}...)
	if err != nil {
		return err
	}

	// TODO: Support alternative load strategy.
	// Currently, this method first loads a new model, and then release the old one.
	// However, releasing the old model before loading the new model is sometimes
	// required to reduce memory consumption. It should be configurable.

	// Exchange instance in `s` when Load succeeded
	s.set(ins, saved.ModulePath, saved.ModuleName, saved.ClassName, saved.BatchSize)
	return nil
}

// PyMLFit fits buckets. fit algorithm and return value is depends on Python
// implementation.
func PyMLFit(ctx *core.Context, stateName string, bucket []data.Map) (data.Value, error) {
	s, err := lookupPyMLState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	return s.FitMap(ctx, bucket)
}

// PyMLPredict predicts data and return estimate value.
func PyMLPredict(ctx *core.Context, stateName string, dt data.Value) (data.Value, error) {
	s, err := lookupPyMLState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	return s.Predict(ctx, dt)
}

func lookupPyMLState(ctx *core.Context, stateName string) (*PyMLState, error) {
	st, err := ctx.SharedStates.Get(stateName)
	if err != nil {
		return nil, err
	}

	if s, ok := st.(*PyMLState); ok {
		return s, nil
	}

	return nil, fmt.Errorf("state '%v' isn't a PyMLState", stateName)
}

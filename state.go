package pymlstate

import (
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/ugorji/go/codec"
	"gopkg.in/sensorbee/py.v0/pystate"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"io"
	"sync"
)

var (
	datPath = data.MustCompilePath("data")
)

// State is python instance specialized to multiple layer classification.
// The python instance and this struct must not be coppied directly by assignment
// statement because it doesn't increase reference count of instance.
type State struct {
	base   *pystate.Base
	params MLParams
	bucket []data.Value
	rwm    sync.RWMutex
}

// MLParams is parameters pymlstate defines in addition to those pystate does.
// These parameters come from a WITH clause of a CREATE STATE statement.
type MLParams struct {
	// BatchSize is number of tuples in a single batch training. Write method,
	// which is usually called by an INSERT INTOT statement via uds Sink, stores
	// tuples without training until it has tuples as many as batch_train_size.
	// This is an optional parameter and its default value is 10.
	BatchSize int `codec:"batch_train_size"`
}

// New creates `core.SharedState` for multiple layer classification.
func New(baseParams *pystate.BaseParams, mlParams *MLParams, params data.Map) (*State, error) {
	b, err := pystate.NewBase(baseParams, params)
	if err != nil {
		return nil, err
	}

	s := &State{
		base:   b,
		params: *mlParams,
		bucket: make([]data.Value, 0, mlParams.BatchSize),
	}
	return s, nil
}

// Terminate terminates this state.
func (s *State) Terminate(ctx *core.Context) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if err := s.base.Terminate(ctx); err != nil {
		return err
	}
	// Don't set s.base = nil because it's used for the termination detection.
	s.bucket = nil
	return nil
}

// Write stores a tuple to its bucket and calls "fit" function every
// "batch_train_size" times.
func (s *State) Write(ctx *core.Context, t *core.Tuple) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if err := s.base.CheckTermination(); err != nil {
		return err
	}

	dataSet, err := t.Data.Get(datPath)
	if err != nil {
		return err
	}

	if s.params.BatchSize > 1 {
		s.bucket = append(s.bucket, dataSet)
		if len(s.bucket) < s.params.BatchSize {
			return nil
		}
	} else {
		if dataSet.Type() == data.TypeArray {
			arr, _ := data.AsArray(dataSet)
			s.bucket = arr
		} else {
			s.bucket = []data.Value{dataSet}
		}
	}

	_, err = s.fit(ctx, s.bucket)
	prevBucketSize := len(s.bucket)
	s.bucket = s.bucket[:0] // clear slice but keep capacity
	if err != nil {
		ctx.ErrLog(err).WithField("bucket_size", prevBucketSize).
			Error("pymlstate's training via Write (INSERT INTO) failed")
		return err
	}

	return nil
}

// Fit receives `data.Array` type but it assumes `[]data.Map` type
// for passing arguments to `fit` method.
func (s *State) Fit(ctx *core.Context, bucket []data.Value) (data.Value, error) {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	return s.fit(ctx, bucket)
}

// fit is the internal implementation of Fit. fit doesn't acquire the lock nor
// check s.ins == nil. RLock is sufficient when calling this method because
// this method itself doesn't change any field of State. Although the model
// will be updated by the data, the model is protected by Python's GIL. So,
// this method doesn't require a write lock.
func (s *State) fit(ctx *core.Context, bucket []data.Value) (data.Value, error) {
	return s.base.Call("fit", data.Array(bucket))
}

// Predict applies the model to the data. It returns a result returned from
// Python script.
func (s *State) Predict(ctx *core.Context, dt data.Value) (data.Value, error) {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	return s.base.Call("predict", dt)
}

// Save saves the model of the state. pystate calls `save` method and
// use its return value as dumped model.
func (s *State) Save(ctx *core.Context, w io.Writer, params data.Map) error {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	if err := s.base.CheckTermination(); err != nil {
		return err
	}

	if err := s.saveState(w); err != nil {
		return err
	}
	return s.base.Save(ctx, w, params)
}

const (
	pyMLStateFormatVersion uint8 = 1
)

func (s *State) saveState(w io.Writer) error {
	if _, err := w.Write([]byte{pyMLStateFormatVersion}); err != nil {
		return err
	}

	// Save parameter of State before save python's model
	msgpackHandle := &codec.MsgpackHandle{}
	var out []byte
	enc := codec.NewEncoderBytes(&out, msgpackHandle)
	if err := enc.Encode(&s.params); err != nil {
		return err
	}

	// Write size of MLParams
	dataSize := uint32(len(out))
	err := binary.Write(w, binary.LittleEndian, dataSize)
	if err != nil {
		return err
	}

	// Write MLParams in msgpack
	n, err := w.Write(out)
	if err != nil {
		return err
	}

	if n < len(out) {
		return errors.New("cannot save the MLParams data")
	}

	return nil
}

// Load loads the model of the state. pystate calls `load` method and
// pass to the model data by using method parameter.
func (s *State) Load(ctx *core.Context, r io.Reader, params data.Map) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if err := s.base.CheckTermination(); err != nil {
		return err
	}
	return s.load(ctx, r, params)
}

func (s *State) load(ctx *core.Context, r io.Reader, params data.Map) error {
	var formatVersion uint8
	if err := binary.Read(r, binary.LittleEndian, &formatVersion); err != nil {
		return err
	}

	// TODO: remove MLParams specific parameters from params

	switch formatVersion {
	case 1:
		return s.loadMLParamsAndDataV1(ctx, r, params)
	default:
		return fmt.Errorf("unsupported format version of State container: %v", formatVersion)
	}
}

func (s *State) loadMLParamsAndDataV1(ctx *core.Context, r io.Reader, params data.Map) error {
	var dataSize uint32
	if err := binary.Read(r, binary.LittleEndian, &dataSize); err != nil {
		return err
	}
	if dataSize == 0 {
		return errors.New("size of MLParams must be greater than 0")
	}

	// Read MLParams from reader
	buf := make([]byte, dataSize)
	n, err := r.Read(buf)
	if err != nil {
		return err
	}
	if n != int(dataSize) {
		return errors.New("read size is different from the size of MLParams")
	}

	// Desirialize MLParams
	var saved MLParams
	msgpackHandle := &codec.MsgpackHandle{}
	dec := codec.NewDecoderBytes(buf, msgpackHandle)
	if err := dec.Decode(&saved); err != nil {
		return err
	}

	if s.base == nil { // loading for the first time
		s.base, err = pystate.LoadBase(ctx, r, params)
		if err != nil {
			return err
		}

	} else {
		if err := s.base.Load(ctx, r, params); err != nil {
			return err
		}
	}
	s.params = saved
	return nil
}

// Fit trains the model. It applies tuples that bucket has in a batch manner.
// The return value of this function depends on the implementation of Python
// UDS.
func Fit(ctx *core.Context, stateName string, bucket []data.Value) (data.Value, error) {
	s, err := lookupState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	return s.Fit(ctx, bucket)
}

// Predict applies the model to the given data and returns estimated values.
// The format of the return value depends on each Python UDS.
func Predict(ctx *core.Context, stateName string, dt data.Value) (data.Value, error) {
	s, err := lookupState(ctx, stateName)
	if err != nil {
		return nil, err
	}

	return s.Predict(ctx, dt)
}

// Flush pymlstate bucket. A return value is always nil.
func Flush(ctx *core.Context, stateName string) (data.Value, error) {
	s, err := lookupState(ctx, stateName)
	if err != nil {
		return nil, err
	}
	s.bucket = s.bucket[:0]
	return nil, nil
}

func lookupState(ctx *core.Context, stateName string) (*State, error) {
	st, err := ctx.SharedStates.Get(stateName)
	if err != nil {
		return nil, err
	}

	if s, ok := st.(*State); ok {
		return s, nil
	}

	return nil, fmt.Errorf("state '%v' isn't a State", stateName)
}

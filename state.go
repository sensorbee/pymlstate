package pymlstate

import (
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/ugorji/go/codec"
	"io"
	"pfi/sensorbee/py/pystate"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"sync"
)

var (
	lossPath = data.MustCompilePath("loss")
	accPath  = data.MustCompilePath("accuracy")
)

// PyMLState is python instance specialized to multiple layer classification.
// The python instance and this struct must not be coppied directly by assignment
// statement because it doesn't increase reference count of instance.
type PyMLState struct {
	base   *pystate.Base
	params MLParams
	bucket data.Array
	rwm    sync.RWMutex
}

// MLParams is parameters required for
type MLParams struct {
	BatchSize int `codec:"batch_size"`
}

// New creates `core.SharedState` for multiple layer classification.
func New(baseParams *pystate.BaseParams, mlParams *MLParams, params data.Map) (*PyMLState, error) {
	b, err := pystate.NewBase(baseParams, params)
	if err != nil {
		return nil, err
	}

	s := &PyMLState{
		base:   b,
		params: *mlParams,
		bucket: make(data.Array, 0, mlParams.BatchSize),
	}
	return s, nil
}

// Terminate this state.
func (s *PyMLState) Terminate(ctx *core.Context) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if err := s.base.Terminate(ctx); err != nil {
		return err
	}
	// Don't set s.base = nil because it's used for the termination detection.
	s.bucket = nil
	return nil
}

// Write and call "fit" function. Tuples is cached per train batch size.
func (s *PyMLState) Write(ctx *core.Context, t *core.Tuple) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if err := s.base.CheckTermination(); err != nil {
		return err
	}

	s.bucket = append(s.bucket, t.Data)
	if len(s.bucket) < s.params.BatchSize {
		return nil
	}

	m, err := s.fit(ctx, s.bucket)
	prevBucketSize := len(s.bucket)
	s.bucket = s.bucket[:0] // clear slice but keep capacity
	if err != nil {
		ctx.ErrLog(err).WithField("bucket_size", prevBucketSize).
			Error("pymlstate's training via Write (INSERT INTO) failed")
		return err
	}

	// TODO: add option to toggle the following logging

	ret, err := data.AsMap(m)
	if err != nil {
		// The following log is optional. So, it isn't a error even if the
		// result doesn't have accuracy and loss fields.
		// TODO: write a warning log after the logging option is added.
		return nil
	}

	var loss float64
	if l, e := ret.Get(lossPath); e != nil {
		// TODO: add warning
		return nil
	} else if loss, e = data.ToFloat(l); e != nil {
		// TODO: add warning
		return nil
	}

	var acc float64
	if a, e := ret.Get(accPath); e != nil {
		// TODO: add warning
		return nil
	} else if acc, e = data.ToFloat(a); e != nil {
		// TODO: add warning
		return nil
	}
	ctx.Log().Debugf("loss=%.3f acc=%.3f", loss/float64(s.params.BatchSize),
		acc/float64(s.params.BatchSize))
	return nil
}

// Fit receives `data.Array` type but it assumes `[]data.Map` type
// for passing arguments to `fit` method.
func (s *PyMLState) Fit(ctx *core.Context, bucket data.Array) (data.Value, error) {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	return s.fit(ctx, bucket)
}

// fit is the internal implementation of Fit. fit doesn't acquire the lock nor
// check s.ins == nil. RLock is sufficient when calling this method because
// this method itself doesn't change any field of PyMLState. Although the model
// will be updated by the data, the model is protected by Python's GIL. So,
// this method doesn't require a write lock.
func (s *PyMLState) fit(ctx *core.Context, bucket data.Array) (data.Value, error) {
	return s.base.Call("fit", bucket)
}

// FitMap receives `[]data.Map`, these maps are converted to `data.Array`
func (s *PyMLState) FitMap(ctx *core.Context, bucket []data.Map) (data.Value, error) {
	args := make(data.Array, len(bucket))
	for i, v := range bucket {
		args[i] = v
	}

	s.rwm.RLock()
	defer s.rwm.RUnlock()
	return s.base.Call("fit", args)
}

// Predict applies the model to the data. It returns a result returned from
// Python script.
func (s *PyMLState) Predict(ctx *core.Context, dt data.Value) (data.Value, error) {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	return s.base.Call("predict", dt)
}

// Save saves the model of the state. pystate calls `save` method and
// use its return value as dumped model.
func (s *PyMLState) Save(ctx *core.Context, w io.Writer, params data.Map) error {
	s.rwm.RLock()
	defer s.rwm.RUnlock()
	if err := s.base.CheckTermination(); err != nil {
		return err
	}

	if err := s.savePyMLMsgpack(w); err != nil {
		return err
	}
	return s.base.Save(ctx, w, params)
}

const (
	pyMLStateFormatVersion uint8 = 1
)

func (s *PyMLState) savePyMLMsgpack(w io.Writer) error {
	if _, err := w.Write([]byte{pyMLStateFormatVersion}); err != nil {
		return err
	}

	// Save parameter of PyMLState before save python's model
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
func (s *PyMLState) Load(ctx *core.Context, r io.Reader, params data.Map) error {
	s.rwm.Lock()
	defer s.rwm.Unlock()
	if err := s.base.CheckTermination(); err != nil {
		return err
	}

	var formatVersion uint8
	if err := binary.Read(r, binary.LittleEndian, &formatVersion); err != nil {
		return err
	}

	// TODO: remove MLParams specific parameters from params

	switch formatVersion {
	case 1:
		return s.loadMLParamsAndDataV1(ctx, r, params)
	default:
		return fmt.Errorf("unsupported format version of PyMLState container: %v", formatVersion)
	}
}

func (s *PyMLState) loadMLParamsAndDataV1(ctx *core.Context, r io.Reader, params data.Map) error {
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
	if err := s.base.Load(ctx, r, params); err != nil {
		return err
	}
	s.params = saved
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

package pymlstate

import (
	"fmt"
	"gopkg.in/sensorbee/py.v0/pystate"
	"gopkg.in/sensorbee/sensorbee.v0/bql/udf"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"io"
)

var (
	batchTrainSizePath = data.MustCompilePath("batch_train_size")
)

// StateCreator is used by BQL to create or load Multiple Layer Classification
// State as a UDS.
type StateCreator struct {
}

var _ udf.UDSLoader = &StateCreator{}

// CreateState creates `core.SharedState`. Some parameters are from pystate
// package. See the document of pystate.BaseParams for details. pymlstate has
// its own parameters, which is defined at MLParams.
func (c *StateCreator) CreateState(ctx *core.Context, params data.Map) (
	core.SharedState, error) {
	bp, err := pystate.ExtractBaseParams(params, true)
	if err != nil {
		return nil, err
	}

	// TODO: extract this code block as ExtractMLParams function
	batchSize := 10
	if bs, err := params.Get(batchTrainSizePath); err == nil {
		var batchSize64 int64
		if batchSize64, err = data.AsInt(bs); err != nil {
			return nil, err
		}
		if batchSize64 <= 0 {
			return nil, fmt.Errorf("batch_train_size must be greater than 0")
		}
		batchSize = int(batchSize64)
		delete(params, "batch_train_size")
	}

	return New(bp, &MLParams{BatchSize: batchSize}, params)
}

// LoadState is same as CREATE STATE.
func (c *StateCreator) LoadState(ctx *core.Context, r io.Reader, params data.Map) (
	core.SharedState, error) {
	s := &State{}
	if err := s.load(ctx, r, params); err != nil {
		return nil, err
	}
	return s, nil
}

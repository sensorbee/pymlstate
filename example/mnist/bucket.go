package mnist

import (
	"fmt"
	"pfi/sensorbee/sensorbee/bql/udf"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"sync"
	"time"
)

var (
	bucketSizePath = data.MustCompilePath("bucket_size")
)

// NewBucketState creates a shared state, which manages a bucket to train &
// predict.
//
// bucket_size: a bucket size, required
func NewBucketState(ctx *core.Context, params data.Map) (core.SharedState, error) {
	bucketSize := 0
	if bs, err := params.Get(bucketSizePath); err != nil {
		return nil, err
	} else if size, err := data.AsInt(bs); err != nil {
		return nil, err
	} else {
		bucketSize = int(size)
	}

	return &dataBucket{
		bucketSize: bucketSize,
		pool:       make(data.Array, 0, bucketSize),
	}, nil
}

type dataBucket struct {
	bucketSize int

	pool   data.Array
	bucket data.Array
	rwm    sync.RWMutex
}

func (b *dataBucket) store(dt data.Value) bool {
	b.pool = append(b.pool, dt)
	if len(b.pool) < b.bucketSize {
		return false
	}
	b.bucket = b.pool
	b.pool = b.pool[:0] // clear slice but keep capacity
	return true
}

func (b *dataBucket) stream() data.Array {
	return b.bucket
}

func (b *dataBucket) Terminate(ctx *core.Context) error {
	return nil
}

// CreateBucketStoreUDSF returns UDSF to store tuple data to the target
// bucket.
//
// stream:     target stream name
// bucketName: target bucket (shared state) name
func CreateBucketStoreUDSF(ctx *core.Context, decl udf.UDSFDeclarer, stream,
	bucketName string) (udf.UDSF, error) {
	if err := decl.Input(stream, &udf.UDSFInputConfig{
		InputName: "mnist_batch",
	}); err != nil {
		return nil, err
	}

	return &bucketStoreUDSF{
		bucketName: bucketName,
	}, nil
}

type bucketStoreUDSF struct {
	bucketName string
}

func (sf *bucketStoreUDSF) Process(ctx *core.Context, t *core.Tuple,
	w core.Writer) error {
	b, err := lookupDataBucketState(ctx, sf.bucketName)
	if err != nil {
		return err
	}

	b.rwm.Lock()
	defer b.rwm.Unlock()

	if !b.store(t.Data) {
		return nil
	}

	now := time.Now()
	m := data.Map{
		"bucket": b.stream(),
	}
	traces := []core.TraceEvent{}
	if len(t.Trace) > 0 {
		traces = make([]core.TraceEvent, len(t.Trace), (cap(t.Trace)+1)*2)
		copy(traces, t.Trace)
	}
	tu := &core.Tuple{
		Data:          m,
		Timestamp:     t.Timestamp,
		ProcTimestamp: now,
		Trace:         traces,
	}
	return w.Write(ctx, tu)
}

func (sf *bucketStoreUDSF) Terminate(ctx *core.Context) error {
	return nil
}

func lookupDataBucketState(ctx *core.Context, name string) (*dataBucket, error) {
	st, err := ctx.SharedStates.Get(name)
	if err != nil {
		return nil, err
	}

	if s, ok := st.(*dataBucket); ok {
		return s, nil
	}
	return nil, fmt.Errorf("state '%v' cannot be converted to data_bucket.state",
		name)
}

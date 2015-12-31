package mnist

import (
	"fmt"
	"math/rand"
	"pfi/sensorbee/sensorbee/bql/udf"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"time"
)

// CreateEpochUDSF returns bucket to batch stream function.
//
// stream:    target stream name
// arrayKey:  key name of splitting array
// batchSize: splitting size
// epoch:     loop count number
// random     randomize before streaming
func CreateEpochUDSF(ctx *core.Context, decl udf.UDSFDeclarer, stream,
	arrayKey string, batchSize, epoch int, random bool) (udf.UDSF, error) {
	if err := decl.Input(stream, &udf.UDSFInputConfig{
		InputName: "mnist_epoch",
	}); err != nil {
		return nil, err
	}

	if arrayKey == "" {
		return nil, fmt.Errorf("'array key name' must not be empty")
	}
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be more than zero")
	}
	if epoch <= 0 {
		return nil, fmt.Errorf("epoch size must be more than zero")
	}

	arrayKeyPath, err := data.CompilePath(arrayKey)
	if err != nil {
		return nil, err
	}
	return &epochUDSF{
		arrayKeyPath: arrayKeyPath,
		batchSize:    batchSize,
		epoch:        epoch,
		random:       random,
	}, nil
}

type epochUDSF struct {
	arrayKeyPath data.Path
	batchSize    int
	epoch        int
	random       bool
}

func (sf *epochUDSF) Process(ctx *core.Context, t *core.Tuple,
	w core.Writer) error {
	var bucket data.Array
	if target, err := t.Data.Get(sf.arrayKeyPath); err != nil {
		return err
	} else if bucket, err = data.AsArray(target); err != nil {
		return err
	}

	bucketNum := len(bucket)
	n := bucketNum / sf.batchSize
	mod := bucketNum % sf.batchSize
	batchNum := n
	if mod != 0 {
		batchNum++
	}

	perm := make([]int, bucketNum, bucketNum)
	for i := range perm {
		perm[i] = i
	}

	traceCopyFlag := len(t.Trace) > 0
	for i := 0; i < sf.epoch; i++ {
		// create randomized indication array list
		// [0..100]
		// -> randomized [9,88, ... , 3]
		// -> separated  [[9,88...],[..3]]
		ind := make([]int, bucketNum, bucketNum)
		copy(ind, perm)
		if sf.random {
			randomPermutaion(ind)
		}
		inds := make([][]int, batchNum, batchNum)
		for j := 0; j < n; j++ {
			temp := ind[j*sf.batchSize : (j+1)*sf.batchSize]
			inds[j] = temp
		}
		if mod != 0 {
			temp := ind[bucketNum-mod : bucketNum]
			inds[batchNum-1] = temp
		}

		for _, in := range inds {
			d := make(data.Array, len(in), len(in))
			for k, p := range in {
				d[k] = bucket[p]
			}
			now := time.Now()
			m := data.Map{
				"epoch": data.Int(i + 1),
				"data":  d,
			}
			traces := []core.TraceEvent{}
			if traceCopyFlag {
				traces = make([]core.TraceEvent, len(t.Trace), (cap(t.Trace)+1)*2)
				copy(traces, t.Trace)
			}
			tu := &core.Tuple{
				Data:          m,
				Timestamp:     now,
				ProcTimestamp: t.ProcTimestamp,
				Trace:         traces,
			}
			w.Write(ctx, tu)
		}
		ctx.Log().Infof("epoch:%d has been emitted", i+1)
	}

	return nil
}

func randomPermutaion(perm []int) {
	for i := range perm {
		j := rand.Intn(i + 1)
		perm[i], perm[j] = perm[j], perm[i]
	}
}

func (sf *epochUDSF) Terminate(ctx *core.Context) error {
	return nil
}

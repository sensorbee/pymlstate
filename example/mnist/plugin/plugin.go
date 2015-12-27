package plugin

import (
	"pfi/sensorbee/pymlstate/example/mnist"
	"pfi/sensorbee/sensorbee/bql"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	bql.MustRegisterGlobalSourceCreator("mnist_source", &mnist.DataSourceCreator{})

	udf.MustRegisterGlobalUDSCreator("mnist_bucket",
		udf.UDSCreatorFunc(mnist.NewBucketState))
	udf.MustRegisterGlobalUDSFCreator("mnist_batch",
		udf.MustConvertToUDSFCreator(mnist.CreateBucketStoreUDSF))
}

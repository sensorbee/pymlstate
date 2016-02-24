package plugin

import (
	"gopkg.in/sensorbee/pymlstate.v0/example/mnist"
	"gopkg.in/sensorbee/sensorbee.v0/bql"
	"gopkg.in/sensorbee/sensorbee.v0/bql/udf"
)

func init() {
	bql.MustRegisterGlobalSourceCreator("mnist_source", &mnist.DataSourceCreator{})

	udf.MustRegisterGlobalUDSCreator("mnist_bucket",
		udf.UDSCreatorFunc(mnist.NewBucketState))
	udf.MustRegisterGlobalUDSFCreator("mnist_batch",
		udf.MustConvertToUDSFCreator(mnist.CreateBucketStoreUDSF))

	udf.MustRegisterGlobalUDSFCreator("mnist_epoch",
		udf.MustConvertToUDSFCreator(mnist.CreateEpochUDSF))
}

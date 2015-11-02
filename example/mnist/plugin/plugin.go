package plugin

import (
	"pfi/sensorbee/pymlstate/example/mnist"
	"pfi/sensorbee/sensorbee/bql"
)

func init() {
	bql.MustRegisterGlobalSourceCreator("mnist_source", &mnist.DataSourceCreator{})
}

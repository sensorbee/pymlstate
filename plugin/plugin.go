package plugin

import (
	"gopkg.in/sensorbee/pymlstate.v0"
	"gopkg.in/sensorbee/sensorbee.v0/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("pymlstate", &pymlstate.StateCreator{})

	udf.MustRegisterGlobalUDF("pymlstate_fit",
		udf.MustConvertGeneric(pymlstate.Fit))
	udf.MustRegisterGlobalUDF("pymlstate_predict",
		udf.MustConvertGeneric(pymlstate.Predict))
}

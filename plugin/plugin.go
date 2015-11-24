package plugin

import (
	"pfi/sensorbee/pymlstate"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("pymlstate", &pymlstate.StateCreator{})

	udf.MustRegisterGlobalUDF("pymlstate_fit",
		udf.MustConvertGeneric(pymlstate.Fit))
	udf.MustRegisterGlobalUDF("pymlstate_fitmap",
		udf.MustConvertGeneric(pymlstate.FitMap))
	udf.MustRegisterGlobalUDF("pymlstate_predict",
		udf.MustConvertGeneric(pymlstate.Predict))
}

package plugin

import (
	"pfi/sensorbee/pymlstate"
	"pfi/sensorbee/sensorbee/bql/udf"
)

func init() {
	udf.MustRegisterGlobalUDSCreator("pymlstate", &pymlstate.PyMLStateCreator{})

	udf.MustRegisterGlobalUDF("pymlstate_fit",
		udf.MustConvertGeneric(pymlstate.PyMLFit))
	udf.MustRegisterGlobalUDF("pymlstate_predict",
		udf.MustConvertGeneric(pymlstate.PyMLPredict))
}

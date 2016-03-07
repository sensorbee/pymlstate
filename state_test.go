package pymlstate

import (
	. "github.com/smartystreets/goconvey/convey"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"testing"
)

func TestPyMLStateFlush(t *testing.T) {
	Convey("Given a context set dummy state", t, func() {
		bu := []data.Value{data.String("a"), data.String("b")}
		cc := &core.ContextConfig{}
		ctx := core.NewContext(cc)
		s := &State{
			bucket: bu,
		}
		stateName := "test_state_for_flush"
		err := ctx.SharedStates.Add(stateName, stateName, s)
		So(err, ShouldBeNil)
		So(len(s.bucket), ShouldEqual, 2)
		Convey("When call flush", func() {
			ac, err := Flush(ctx, stateName)
			So(ac, ShouldBeNil)
			So(err, ShouldBeNil)
			Convey("Then state bucket should be empty", func() {
				So(len(s.bucket), ShouldEqual, 0)
			})
		})
	})
}

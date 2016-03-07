package pymlstate

import (
	. "github.com/smartystreets/goconvey/convey"
	"gopkg.in/sensorbee/py.v0/pystate"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"testing"
)

func TestPyMLStateFitAndPredict(t *testing.T) {
	cc := &core.ContextConfig{}
	ctx := core.NewContext(cc)
	Convey("Given a context set pymlstate for fit/predict test", t, func() {
		baseParams := &pystate.BaseParams{
			ModulePath: "./",
			ModuleName: "_test_pymlstate",
			ClassName:  "TestClass",
		}
		mlParams := &MLParams{
			BatchSize: 10,
		}

		s, err := New(baseParams, mlParams, data.Map{})
		So(err, ShouldBeNil)
		Reset(func() {
			s.Terminate(ctx)
		})
		err = ctx.SharedStates.Add("pystate_test", "py", s)
		So(err, ShouldBeNil)
		Convey("When call fit", func() {
			bu := []data.Value{
				data.String("a"), data.String("b"),
			}
			ac, err := Fit(ctx, "pystate_test", bu)
			So(err, ShouldBeNil)
			Convey("Then fit function should be called", func() {
				So(ac, ShouldEqual, "fit called")

				Convey("And when call predict", func() {
					ac2, err := Predict(ctx, "pystate_test", data.String("c"))
					So(err, ShouldBeNil)
					Convey("Then predict function should be called", func() {
						So(ac2, ShouldEqual, "predict called")
					})
				})
			})
		})
	})
}

func TestPyMLStateWrite(t *testing.T) {
	cc := &core.ContextConfig{}
	ctx := core.NewContext(cc)
	Convey("Given a context set pymlstate for write test", t, func() {
		baseParams := &pystate.BaseParams{
			ModulePath: "./",
			ModuleName: "_test_pymlstate",
			ClassName:  "TestClass",
		}
		mlParams := &MLParams{
			BatchSize: 3,
		}

		s, err := New(baseParams, mlParams, data.Map{})
		So(err, ShouldBeNil)
		Reset(func() {
			s.Terminate(ctx)
		})
		err = ctx.SharedStates.Add("pystate_test", "py", s)
		So(err, ShouldBeNil)
		Convey("When write a data", func() {
			dt := data.String("1")
			tu := &core.Tuple{
				Data: data.Map{
					"data": dt,
				},
			}
			err := s.Write(ctx, tu)
			So(err, ShouldBeNil)
			Convey("Then fit function should not be called", func() {
				ac, err := s.base.Call("confirm_to_call_fit")
				So(err, ShouldBeNil)
				So(ac, ShouldEqual, 0)
				So(len(s.bucket), ShouldEqual, 1)

				Convey("And when write data until bucket size", func() {
					tu2 := tu.Copy()
					err := s.Write(ctx, tu2)
					So(err, ShouldBeNil)
					tu3 := tu.Copy()
					err = s.Write(ctx, tu3)
					So(err, ShouldBeNil)
					Convey("Then fit function should be called and bucket is flushed", func() {
						ac2, err := s.base.Call("confirm_to_call_fit")
						So(err, ShouldBeNil)
						So(ac2, ShouldEqual, 1)
						So(len(s.bucket), ShouldEqual, 0)
					})
				})
			})
		})
	})
}

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

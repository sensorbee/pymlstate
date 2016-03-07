package pymlstate

import (
	"bytes"
	. "github.com/smartystreets/goconvey/convey"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"testing"
)

func TestCreatePyMLState(t *testing.T) {
	ctx := &core.Context{}
	Convey("Given a state creator", t, func() {
		sc := StateCreator{}
		Convey("When create a pymlstate with empty parameter", func() {
			params := data.Map{}
			_, err := sc.CreateState(ctx, params)
			Convey("Then creator should return an error", func() {
				So(err, ShouldNotBeNil)
			})
		})

		Convey("When create a pymlstate with base parameters", func() {
			params := data.Map{
				"module_path": data.String("./"),
				"module_name": data.String("_test_pymlstate"),
				"class_name":  data.String("TestClass"),
			}
			s, err := sc.CreateState(ctx, params)
			So(err, ShouldBeNil)
			Reset(func() {
				s.Terminate(ctx)
			})
			Convey("Then the state should be set up with default parameter", func() {
				ps, ok := s.(*State)
				So(ok, ShouldBeTrue)
				So(ps.params.BatchSize, ShouldEqual, 1)
			})
		})

		Convey("When create a pymlstate with customized parameters", func() {
			params := data.Map{
				"module_path":      data.String("./"),
				"module_name":      data.String("_test_pymlstate"),
				"class_name":       data.String("TestClass"),
				"batch_train_size": data.Int(50),
			}
			s, err := sc.CreateState(ctx, params)
			So(err, ShouldBeNil)
			Reset(func() {
				s.Terminate(ctx)
			})
			Convey("Then the state should be set up with customized parameter", func() {
				ps, ok := s.(*State)
				So(ok, ShouldBeTrue)
				So(ps.params.BatchSize, ShouldEqual, 50)
				So(len(ps.bucket), ShouldEqual, 0)
				So(cap(ps.bucket), ShouldEqual, 50)
			})
		})
	})
}

func TestLoadPyMLState(t *testing.T) {
	cc := &core.ContextConfig{}
	ctx := core.NewContext(cc)
	Convey("Given a state creator", t, func() {
		sc := StateCreator{}
		Convey("When create a pymlstate for confirm loading", func() {
			params := data.Map{
				"module_path":      data.String("./"),
				"module_name":      data.String("_test_pymlstate"),
				"class_name":       data.String("TestClass"),
				"batch_train_size": data.Int(50),
			}
			s, err := sc.CreateState(ctx, params)
			So(err, ShouldBeNil)
			Reset(func() {
				s.Terminate(ctx)
			})

			err = ctx.SharedStates.Add("test_pymlstate_load", "py", s)
			So(err, ShouldBeNil)
			Convey("And when save the state", func() {
				ps, ok := s.(*State)
				So(ok, ShouldBeTrue)
				buf := bytes.NewBuffer(nil)
				err := ps.Save(ctx, buf, data.Map{})
				So(err, ShouldBeNil)

				Convey("And when load the state", func() {
					s2, err := sc.LoadState(ctx, buf, data.Map{})
					So(err, ShouldBeNil)
					Reset(func() {
						s2.Terminate(ctx)
					})
					Convey("Then the state should be loaded validly", func() {
						ps2, ok := s2.(*State)
						So(ok, ShouldBeTrue)
						So(ps2.params.BatchSize, ShouldEqual, 50)
					})
				})
			})
		})
	})
}

package mnist

import (
	. "github.com/smartystreets/goconvey/convey"
	"pfi/sensorbee/sensorbee/core"
	"pfi/sensorbee/sensorbee/data"
	"testing"
)

func TestEpochUDSFProcess(t *testing.T) {
	cc := &core.ContextConfig{}
	ctx := core.NewContext(cc)
	Convey("Given a bucketToBatch UDSF", t, func() {
		sf := epochUDSF{
			arrayKeyPath: data.MustCompilePath("key"),
			batchSize:    3,
			epoch:        2,
			random:       true,
		}
		Convey("When process empty tuple", func() {
			tu := &core.Tuple{
				Data: data.Map{},
			}
			w := core.WriterFunc(func(ctx *core.Context, t *core.Tuple) error {
				return nil
			})
			err := sf.Process(ctx, tu, w)
			Convey("Then the UDSF should return an error", func() {
				So(err, ShouldNotBeNil)
			})
		})

		Convey("When process empty array", func() {
			tu := &core.Tuple{
				Data: data.Map{
					"key": data.Array{},
				},
			}
			i := 0
			w := core.WriterFunc(func(ctx *core.Context, t *core.Tuple) error {
				i++
				return nil
			})
			err := sf.Process(ctx, tu, w)
			So(err, ShouldBeNil)
			Convey("Then the UDSF should not generate any stream", func() {
				So(i, ShouldEqual, 0)
			})
		})

		Convey("When process data less than batch size", func() {
			is := data.Array{data.Int(10), data.Int(20)}
			tu := &core.Tuple{
				Data: data.Map{
					"key": is,
				},
			}
			i := 0
			w := core.WriterFunc(func(ctx *core.Context, t *core.Tuple) error {
				d, err := t.Data.Get(data.MustCompilePath("data"))
				So(err, ShouldBeNil)
				da, err := data.AsArray(d)
				So(err, ShouldBeNil)
				So(da, ShouldResemble, is)
				i++
				return nil
			})
			err := sf.Process(ctx, tu, w)
			So(err, ShouldBeNil)
			Convey("Then the UDSF should not generate one stream * epoch", func() {
				So(i, ShouldEqual, 2)
			})
		})

		Convey("When process data over than batch size", func() {
			is := data.Array{data.Int(0), data.Int(1), data.Int(2), data.Int(3),
				data.Int(4), data.Int(5), data.Int(6), data.Int(7), data.Int(8),
				data.Int(9)}
			tu := &core.Tuple{
				Data: data.Map{
					"key": is,
				},
			}
			i := 0
			act := []data.Value{}
			w := core.WriterFunc(func(ctx *core.Context, t *core.Tuple) error {
				d, err := t.Data.Get(data.MustCompilePath("data"))
				So(err, ShouldBeNil)
				da, err := data.AsArray(d)
				So(err, ShouldBeNil)

				ep, err := t.Data.Get(data.MustCompilePath("epoch"))
				So(err, ShouldBeNil)
				epoch, err := data.AsInt(ep)
				So(err, ShouldBeNil)
				So(epoch, ShouldEqual, i/4+1)

				if i%4 == 3 {
					So(len(da), ShouldEqual, 1)
				} else {
					So(len(da), ShouldEqual, 3)
				}
				i++
				act = append(act, da...)
				return nil
			})
			err := sf.Process(ctx, tu, w)
			So(err, ShouldBeNil)
			Convey("Then the UDSF should not generate four stream * epoch", func() {
				So(i, ShouldEqual, 8)
				So(len(act), ShouldEqual, 20)
			})
		})
	})
}

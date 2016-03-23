package mnist

import (
	. "github.com/smartystreets/goconvey/convey"
	"gopkg.in/sensorbee/sensorbee.v0/core"
	"gopkg.in/sensorbee/sensorbee.v0/data"
	"testing"
)

func TestCreateDataBucketState(t *testing.T) {
	cc := &core.ContextConfig{}
	ctx := core.NewContext(cc)
	Convey("Given a state creator", t, func() {

		Convey("When pass an empty parameter", func() {
			params := data.Map{}
			_, err := NewBucketState(ctx, params)
			Convey("Then the creator should not create a state", func() {
				So(err, ShouldNotBeNil)
			})
		})

		Convey("When pass a valid parameter", func() {
			params := data.Map{
				"bucket_size": data.Int(3),
			}
			s, err := NewBucketState(ctx, params)
			So(err, ShouldBeNil)
			Reset(func() {
				s.Terminate(ctx)
			})
			Convey("Then the creator should return a state", func() {
				db, ok := s.(*dataBucket)
				So(ok, ShouldBeTrue)
				So(db.bucketSize, ShouldEqual, 3)
			})
		})
	})
}

func TestBucketStoreUDSFProcess(t *testing.T) {
	cc := &core.ContextConfig{}
	ctx := core.NewContext(cc)
	Convey("Given a bucket which set up three tuples pool & bucket store UDSF", t, func() {
		params := data.Map{
			"bucket_size": data.Int(3),
		}
		ss, err := NewBucketState(ctx, params)
		So(err, ShouldBeNil)
		err = ctx.SharedStates.Add("test_bucket", "bucket_state", ss)
		So(err, ShouldBeNil)

		udsf := bucketStoreUDSF{
			bucketName: "test_bucket",
		}

		Convey("When three tuples are processed", func() {
			var written data.Array
			w := core.WriterFunc(func(ctx *core.Context, t *core.Tuple) error {
				bu, err := t.Data.Get(data.MustCompilePath("bucket"))
				So(err, ShouldBeNil)
				array, err := data.AsArray(bu)
				So(err, ShouldBeNil)
				So(len(array), ShouldEqual, 3)
				written = array

				return nil
			})

			m1 := data.Map{
				"data": data.String("1"),
			}
			t1 := &core.Tuple{
				Data: m1,
			}
			err := udsf.Process(ctx, t1, w)
			So(err, ShouldBeNil)

			m2 := data.Map{
				"data": data.String("2"),
			}
			t2 := &core.Tuple{
				Data: m2,
			}
			err = udsf.Process(ctx, t2, w)
			So(err, ShouldBeNil)

			m3 := data.Map{
				"data": data.String("3"),
			}
			t3 := &core.Tuple{
				Data: m3,
			}
			err = udsf.Process(ctx, t3, w)
			So(err, ShouldBeNil)

			Convey("Then the three tuples should be written", func() {
				expected := data.Array{m1, m2, m3}
				So(written, ShouldResemble, expected)

				Convey("And when more three tuples are processed", func() {

					m4 := data.Map{
						"data": data.String("4"),
					}
					t4 := &core.Tuple{
						Data: m4,
					}
					err := udsf.Process(ctx, t4, w)
					So(err, ShouldBeNil)

					m5 := data.Map{
						"data": data.String("5"),
					}
					t5 := &core.Tuple{
						Data: m5,
					}
					err = udsf.Process(ctx, t5, w)
					So(err, ShouldBeNil)

					m6 := data.Map{
						"data": data.String("6"),
					}
					t6 := &core.Tuple{
						Data: m6,
					}
					err = udsf.Process(ctx, t6, w)
					So(err, ShouldBeNil)

					Convey("Then the more three tuples should be written", func() {
						expected2 := data.Array{m4, m5, m6}
						So(written, ShouldResemble, expected2)
					})
				})
			})
		})
	})
}

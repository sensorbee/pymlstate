Call chainer MNIST sample code from SensorBee.

# Setup

## 1. link python

sensorbee/py plugin is required to link SensorBee and Python Use pkg-config, detail:  [Set up to link python](https://github.com/sensorbee/py#set-up-to-link-python)

sensorbee/py supports only Python 2.x .

## 2. install python library

* chainer
    * support 1.5.x or later

## 3. install build_sensorbee

```bash
cd $GOPATH/src/gopkg.in/sensorbee/sensorbee.v0/cmd/build_sensorbee
go install
```

## 4. make working directory

* uds: For saving/loading SensorBee's shared state, which includes chainer's model.
* data: For MNIST train/test files, and SensorBee puts log file.

```bash
cd $GOPATH/src/gopkg.in/sensorbee/pymlstate.v0/example/mnist/sensorbee
mkdir uds
mkdir data
```

Put MNIST binary data on "data".

* train-images-idx3-ubyte
* train-labels-idx1-ubyte
* t10k-images-idx3-ubyte
* t10k-labels-idx1-ubyte

## 5. build sensorbee

```
build_sensorbee --download-plugins=false
```

### build.yaml

```yaml
--- # SensorBee plug-in list
plugins:
- gopkg.in/sensorbee/py.v0/pystate/plugin
- gopkg.in/sensorbee/pymlstate.v0/plugin
- gopkg.in/sensorbee/pymlstate.v0/example/mnist/plugin
```

# Training

```bash
./sensorbee runfile -c conf.yaml -s ml_mnist -t mnist train.bql
```

"uds/mnist-ml_mnist-default.state" will be created. `loss` values are logged in "data/trained.jsonl" .

# Test

```bash
./sensorbee runfile -c conf.yaml -s ml_mnist -t mnist test.bql
```

`accuracy` values are logged in "data/acc_log.jsonl" .

# [Experiment] Online test

```bash
./sensorbee run -c conf.yaml
```

```bash
curl -v -H "Accept: application/json" -H "Content-type: application/json" -X POST -d "{\"queries\":\"EVAL pymlstate_predict('ml_mnist', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7490196078431373,1,1,0.18823529411764706,0,0,0,0.4980392156862745,1,0.25098039215686274,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5019607843137255,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.25098039215686274,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.7490196078431373,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7490196078431373,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.996078431372549,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7411764705882353,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7490196078431373,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3333333333333333,1,1,0.25098039215686274,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5019607843137255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4980392156862745,0.996078431372549,1,1,1,1,1,0.5019607843137255,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4745098039215686,0.5019607843137255,0.3215686274509804,0,0,0,0,0,0,0,0,0.48627450980392156,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);\"}" http://localhost:8090/api/v1/topologies/mnist/queries
```

Will be returned `{"result":2}`

import six


class TestClass(object):

    @staticmethod
    def create():
        self = TestClass()
        self.cnt = 0
        return self

    @staticmethod
    def load(filepath, *args, **kwargs):
        with open(filepath, 'r') as f:
            return six.moves.cPickle.load(f)

    def fit(self, data):
        self.cnt += 1
        return 'fit called'

    def predict(self, data):
        return 'predict called'

    def save(self, filepath, *args, **kwargs):
        with open(filepath, 'w') as f:
            six.moves.cPickle.dump(self, f)

    def confirm_to_call_fit(self):
        return self.cnt

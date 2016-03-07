import six


class TestClass(object):

    @staticmethod
    def create():
        self = TestClass()
        return self

    @staticmethod
    def load(filepath, *args, **kwargs):
        with open(filepath, 'r') as f:
            return six.moves.cPickle.load(f)

    def save(self, filepath, *args, **kwargs):
        with open(filepath, 'w') as f:
            six.moves.cPickle.dump(self, f)

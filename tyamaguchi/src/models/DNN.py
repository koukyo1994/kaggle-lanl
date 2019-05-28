import chainer
import chainer.functions as F
import chainer.links as L



class LinearBlock(chainer.Chain):

    def __init__(self, n_out=512, drop=False):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, n_out, initialW=w)
            # self.bn = L.BatchNormalization(n_out)
        self.drop = drop

    def __call__(self, x):
        # h = F.relu(self.bn(self.fc(x)))
        h = F.relu(self.fc(x))
        if self.drop:
            h = F.dropout(h, 0.3)
        return h


class DNN(chainer.ChainList):

    def __init__(self):
        super(DNN, self).__init__(
            LinearBlock(256,True),
            LinearBlock(128,True),
            LinearBlock(96,True),
            L.Linear(None, 1)
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x

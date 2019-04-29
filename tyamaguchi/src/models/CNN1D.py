import chainer
import chainer.functions as F
import chainer.links as L


class ConvBlock(chainer.Chain):

    def __init__(self, n_ch, window=5, stride=1, pad=0, dilate=1, pool_drop=False):
        w = chainer.initializers.HeNormal()
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution1D(None, n_ch, window, stride, pad, nobias=True, initialW=w, dilate=dilate)
            self.bn = L.BatchNormalization(n_ch)
        self.pool_drop = pool_drop

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        if self.pool_drop:
            h = F.max_pooling_1d(h, 10, 10)
            h = F.dropout(h, ratio=0.2)
        return h

class LinearBlock(chainer.Chain):

    def __init__(self, drop=False):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, 1024, initialW=w)
            self.bn = L.BatchNormalization(1024)
        self.drop = drop

    def __call__(self, x):
        h = F.relu(self.bn(self.fc(x)))
        if self.drop:
            h = F.dropout(h)
        return h


class CNN1D(chainer.ChainList):

    def __init__(self):
        super(CNN1D, self).__init__(
            ConvBlock(2, 100, 5),
            ConvBlock(4, 50, 5),
            ConvBlock(8, 20, 3),
            ConvBlock(16, 10, pool_drop=True),
            ConvBlock(64, 10, 2),
            ConvBlock(256, 3, 2),
            ConvBlock(512, 3),
            ConvBlock(1024,3,1,1),
            ConvBlock(1024,3,1,1),
            ConvBlock(2048),
            ConvBlock(2048, pool_drop=True),
            LinearBlock(),
            LinearBlock(),
            L.Linear(None, 1)
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x

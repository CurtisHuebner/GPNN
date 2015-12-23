import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class VLinear(Link):
    def __init__(self,ln_sigmaW,ln_sigmaB,muW,muB):
        super(VLinear,self).__init__(
                ln_sigmaW=ln_sigmaW,
                ln_sigmaB=ln_sigmaB,
                muW=muW,
                muB=muB
            )

    def __call__(self,x):
        w = F.gaussian(self.muW,self.ln_sigmaW)
        b = F.gaussian(self.muB,self.ln_sigmaB)
        return F.linear(x,w,b)

class VUniaryT(Link):
    def __init__(self,size):
        pass
a = np.array([1],dtype='float32')
b = np.array([0],dtype='float32')
c = VLinear(a,a,b,b)
x = c(a)
x.backwards()

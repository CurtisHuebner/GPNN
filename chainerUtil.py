import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class VLinear(Link):
    def __init__(self,inSize,outSize,iMtx=False):
        super(VLinear,self).__init__(
                lnSigmaW = (inSize,outSize),
                muW = (inSize,outSize),
                lnSigmaB = (outSize),
                muB = (outSize)
            )
        self.lnSigmaW.data = np.ones((inSize,outSize),dtype='float32')
        self.muW.data = np.ones((inSize,outSize),dtype='float32')
        self.lnSigmaB.data = np.zeros(outSize,dtype='float32')
        self.muB.data = np.zeros(outSize,dtype='float32')


    def __call__(self,x):
        print(self.muW.data.shape)
        print(self.lnSigmaW.data.shape)
        w = F.gaussian(self.muW,self.lnSigmaW)
        b = F.gaussian(self.muB,self.lnSigmaB)
        return F.linear(x,w,b)

class VUniaryT(Chain):
    def __init__(self,size):
        pass

class Branch(Funciton)
    def __init__(control,f,g,x):
        if (control < 0):
            y = f(x)
        else:
            y = g(x)
        controlFactor = control/(control^2 + 1)

class Lookup
    pass

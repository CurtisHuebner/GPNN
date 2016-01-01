import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

#Linear transformation used to preform variational inference
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

        calledValues = dict()
    
    #returns the kl divergence between the model and the prior
    def getDivergence=(p):
        assert(isinstance(VLinear,p))
        #TODO double check that normalizing variance rather than std-dev is allowed
        nW = (self.lnSigmaW - p.lnSigmaW,self.muW - p.muW)
        nB = (self.lnSigmaB - p.lnSigmaB,self.muB - p.muB) 
        f = F.gaussian_kl_divergence

        #compute the sum of the klDivergences between the prior values and the current values
        divergence = F.sum(f(nB[1],nB[0])) + F.sum(f(nW[1],nW[0]))

        return divergence

    def __call__(self,x,seed):
        if (calledValues[seed] = None):
            print(self.muW.data.shape)
            print(self.lnSigmaW.data.shape) 
            w = F.gaussian(self.muW,self.lnSigmaW)
            b = F.gaussian(self.muB,self.lnSigmaB)
            calledValues[seed] = (w,b)
        else
            w,b = calledValues[seed]
        return F.linear(x,w,b)

    def clear()
        calledValues = dict()

class VUniaryT(Chain):
    def __init__(self,size):
        pass

class Branch:
    def __init__(self,f,g):
        self.f = f
        self.g = g
        
    def __call__(control,x):
        if (control < 0):
            y = self.f(x)
        else:
            y = self.g(x)
        c = (1/(control^2 + 1))
        return x*c + (1-c)*y

class DiffArray:
    def __init__(self,exp_elements,size):
        self.elems = 2^exp_elements
        self.exp_elems = exp_elements
        self.size  = size
        self.array = Variable(np.array((self.elems,self.size)))

    def read(address):
        #map from the reals to the hypercube of dimesion n
        index = F.tanh(address)
        
        #map from a point to the nearest corner of the hypercube
        f = lambda x: {if x > 0: True else False}
        mainIndex = np.vectorize(f,index.data,cache=True)

        mainValue = F.select_item(array,lookup(mainIndex))
        scaleFactor =F.exp(F.sum(F.log(F.absolute(x))))

        return mainValue * scaleFactor

    def lookup(index):
        address = 0
        exp = 0;
        for i in range(index.shape[0]):
            if (index[i]):
                address += 2^exp
        exp += 1
        return address

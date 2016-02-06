import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

#Linear transformation used to preform variational inference
class VLinear(Link):
    #TODO finish adding support for nobias
    def __init__(self,inSize,outSize,iMtx=False,initialW=None,initialB=None,nobias=False):
        super(VLinear,self).__init__(
                lnSigmaW = (outSize,inSize),
                muW = (outSize,inSize),
                lnSigmaB = (outSize),
                muB = (outSize)
            )
        self.lnSigmaW.data = np.ones((outSize,inSize),dtype='float32')-7
        self.muW.data = np.ones((outSize,inSize),dtype='float32')
        self.lnSigmaB.data = np.zeros(outSize,dtype='float32')-7
        self.muB.data = np.zeros(outSize,dtype='float32')

        if initialW is not None:
            self.lnSigmaW.data,self.muW.data = initialW
        if initialB is not None:
            self.lnSigmaB.data,self.muB.data = initialB
        
        self.calledValues = dict()
    
    #returns the kl divergence between the model and the prior
    def getDivergence(self,p):
        assert(isinstance(p,VLinear))
        #TODO double check that normalizing variance rather than std-dev is allowed
        nW = (self.lnSigmaW - p.lnSigmaW,self.muW - p.muW)
        nB = (self.lnSigmaB - p.lnSigmaB,self.muB - p.muB)
        f = F.gaussian_kl_divergence

        #compute the sum of the klDivergences between the prior values and the current values
        b = f(nB[1],nB[0])
        w = f(nW[1],nW[0])
        divergence = f(nW[1],nW[0]) + f(nB[1],nB[0])
        return divergence
    
    #Store previously computed values in a hash table for reuse in recurrent networks
    def __call__(self,x,seed):
        if (seed not in self.calledValues):
            w = F.gaussian(self.muW,self.lnSigmaW)
            b = F.gaussian(self.muB,self.lnSigmaB)
            self.calledValues[seed] = (w,b)
        else:
            w,b = self.calledValues[seed]
        return F.linear(x,w,b)

    def clear():
        self.calledValues = dict()

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
        f = lambda x: x > 0
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

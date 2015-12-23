import itertools

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class CharRNN(Chain):
    def __init__(self,charInWidth,lstmSize):
        self.charInWidth = charInWidth,
        super(CharRNN,self).__init__(
            embed = L.EmbedID(charInWidth,lstmSize),
            lstm = L.LSTM(lstmSize,charInWidth)
        )
    
    #x is an integer between 0 and charInWidth
    #embedded in an ChainerVariable
    #
    #the function simulates 1 RNN step and
    #has the side effect of updating the state of the RNN 
    #
    #returns unnormalized log probability distribution 
    #over possible next inputs
    def __call__(self,x):
        iArray = self.embed(x)
        oVec = self.lstm(iArray)
        return oVec

    def reset(self):
        self.lstm.reset_state()

class EvalCRNN(Chain):
    def __init__(self,charInWidth,lstmSize):
        super(EvalCRNN,self).__init__(
            cRNN = CharRNN(charInWidth,lstmSize)
        )
    
    #data is a 1 dimensional numpy array of integers that
    #represents the data used to train the model. 
    #if retainState = false then the rnn is reset
    #before running
    def __call__(self,data,retainState=False):
        if (retainState == False):
            self.cRNN.reset()
        
        #compute loss
        loss = 0 
        for i in range(data.shape[0]-1):
            inVal = Variable(data[i,np.newaxis])
            outVal = self.cRNN(inVal)
            target = Variable(data[i+1,np.newaxis])
            #accumulate loss
            val = F.softmax_cross_entropy(outVal,target)
            loss += val

        return loss

def testCRNN(maxIndex,indicies):
    cRNN = EvalCRNN(maxIndex,10000)
    cRNN.zerograds()
    output = cRNN(indicies[0:100])
    output.backward(retain_grad=True)
    print(output.grad)
    parmeters = cRNN.params()
    for p in parmeters:
        print(p.grad)

def optimizeCRNN(iterNum,maxIndex,indicies):
    model = EvalCRNN(maxIndex,1000)
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    
    print(model(indicies[0:100]).data)
    for i in range(iterNum):
        model.zerograds()
        loss = model(indicies[0:100])
        loss.backward()

        optimizer.update()

        print(model(indicies[0:100]).data)

#Accepts a text file Address as input
#
#returns a (maxIndex,indices) tuple where maxIndex is an integer 
#and indices is a list of numbers between [0,maxIndex)
def fileToIndicies(fileAddress):
    #Open and load data into memory as a string
    f = open('data/mobyDick.txt', 'r')
    data = f.read()
    
    #Convert string into indicies 
    chars = list(set(data))
    indexList = [x for x in range(len(chars))]
    maxIndex = len(chars)

    ctoi = dict(zip(chars,indexList))
    itoc = dict(zip(indexList,chars))

    f = lambda x: ctoi[x]

    indicies = [x for x in map(f,data)]

    return (maxIndex,indicies)


############################################################

(a,b) = fileToIndicies('data/mobyDick.txt')
#testCRNN(a,np.array(b,dtype='int32'))
optimizeCRNN(100,a,np.array(b,dtype='int32'))





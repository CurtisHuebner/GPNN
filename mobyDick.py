import itertools

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, optimizer
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import myLSTM as M

import time


class CharRNN(Chain):
    def __init__(self,charInWidth,rnnSize):
        self.charInWidth = charInWidth,
        super(CharRNN,self).__init__(
            embed = L.EmbedID(charInWidth,rnnSize),
            rnn1 = L.LSTM(rnnSize,rnnSize),
            rnn2 = L.LSTM(rnnSize,rnnSize),
            l = L.Linear(rnnSize,charInWidth)
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
        iVec = self.embed(x)
        y1 = self.rnn1(iVec)
        y2 = self.rnn2(y1)
        oVec = self.l(y2)
        return oVec

    def reset(self):
        self.rnn1.reset_state()
        self.rnn2.reset_state()

class EvalCRNN(Chain):
    def __init__(self,charInWidth,lstmSize):
        super(EvalCRNN,self).__init__(
            cRNN = CharRNN(charInWidth,lstmSize)
        )
    
    #data is a 1 dimensional numpy array of integers that
    #represents the data used to train the model. 
    #if retainState = false then the rnn is reset
    #before running
    def __call__(self,data,retainState=True):
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
    batchSize = 1000
    model = EvalCRNN(maxIndex,500)
    print(len(indicies),computeEntropy(maxIndex,indicies))
    learningRate = 0.001
    epoch = 3 
    for j in range(epoch):
        
        my_optimizer = optimizers.RMSpropGraves(lr = learningRate)
        my_optimizer.setup(model) 
        my_optimizer.add_hook(optimizer.GradientClipping(1))
        
        model.cRNN.reset()
        
        loss = Variable(np.array([[0]]))
        for i in range(iterNum):
            t1 = time.clock()
            model.zerograds()
            loss.unchain_backward()
            loss = model(indicies[batchSize*i:batchSize*(i+1)])
            loss.backward()
            t2 = time.clock()
            
            msg = "iter: " + str(i + iterNum * j + 1) + "/" + str(iterNum * epoch) 
            msgLoss = "loss: " + str(loss.data/batchSize)
            msgNorm = "grad: " + str(my_optimizer.compute_grads_norm())
            msgTime = "time: " + str(t2 - t1) + " seconds"
            print(msgLoss,msgNorm,msg,msgTime)
            my_optimizer.update()

        learningRate *= 0.50

    print(model(indicies[batchSize*(iterNum):batchSize*(iterNum+10)]).data/(batchSize*10))
    return model.cRNN

def sampleCRNN(cRNN,length,itoc,t=1):
    maxIndex = cRNN.charInWidth[0]
    indexList = np.array([x for x in range(maxIndex)],dtype='int32')
    nextChar = np.zeros(1,dtype='int32')
    nextChar[0] = np.random.choice(indexList.flatten())
    charList = []
    for i in range(length):
        logDistribution = cRNN(Variable(nextChar)).data
        distribution = np.exp(logDistribution/t) / np.sum(np.exp(logDistribution/t))
        nextChar[0] = np.random.choice(indexList,p=distribution.flatten())
        charList.append(nextChar[0])

    out = "".join(map((lambda x: itoc[x]),charList))
    return out

def computeEntropy(maxIndex,indicies):
    total = len(indicies)
    counts = np.zeros(maxIndex,dtype='int32')
    for i in indicies:
        counts[i] += 1
    probabilites = np.array(counts,dtype='float64')/total
    entropies = -probabilites * np.log(probabilites)
    return np.sum(entropies)
    

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

    return (maxIndex,indicies,itoc)


############################################################

(a,b,itoc) = fileToIndicies('data/mobyDick.txt')
#testCRNN(a,np.array(b,dtype='int32'))
cRNN = optimizeCRNN(1000,a,np.array(b,dtype='int32'))
print(sampleCRNN(cRNN,5000,itoc))





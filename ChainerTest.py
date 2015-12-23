import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

x_data = np.array(np.ones((2,3))*9, dtype=np.float32)
x = Variable(x_data)

z = 2*x
y = x**7 - z + 1

print(y.data)
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward(retain_grad=True)
print(x.grad)

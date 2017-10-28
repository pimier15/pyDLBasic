import Ooptimizer as opt
import numpy as np
from GraphLayer import *
from collections import OrderedDict
from Common import *
from TwoLayerNet_BackProp import *

from dataset.mnist import load_mnist



np.random.seed(1)
(x_train,y_train) , (x_test,y_test) = load_mnist(True,True,True)

epoch = 200

x_test = x_test[:100]
y_test = y_test[:100]


inputSize = 784
hiddenSize = 50
outputSize = 10




network = Twolayernet_BackProp( inputSize,
                                hiddenSize,
                                outputSize)


predictList = []
Losslist = []
AccList = []

for i in range(epoch):
    res = network.Predict( x_test )    
    grad = network.BGradient(x_test , y_test)





print(res)


















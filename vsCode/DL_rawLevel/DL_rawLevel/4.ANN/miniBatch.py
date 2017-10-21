import numpy as np
from Common import Activation, Diff , SGD , LossFun
from TwoLayerNetmy import Twolayernetmy
import os
import sys
from matplotlib import pyplot as plt

sys.path.append(os.pardir)
sys.path.append(os.pardir+"dataset")

from dataset.mnist import load_mnist

(x_train,y_train) , (x_test,y_test) = load_mnist(True,True)




trainLossList = []

# Setting
iter = 10
trainSize = x_train.shape[0] 
batchSize = 100
lr = 0.1

inputSize = 784
hiddenSize = 50
outputSize = 10

net = Twolayernetmy( 
            inputSize ,
            hiddenSize , 
            outputSize , 
            isOneHot=  False)

for i in range(iter):
    print( "Iter : {0}".format(i) )
    batchMask = np.random.choice(trainSize,batchSize)
    xBatch = x_train[batchMask]
    tBatch = y_train[batchMask]
    
    #Gradient Calculate
    grad = net.NGradient(xBatch,tBatch)

    for key in ('W1','b1','W2','b2'):
        # Update Parameter with SGD
        net.params[key] -= lr * grad[key]

    loss = net.Loss(xBatch,tBatch)
    trainLossList.append(loss)




xlabels = range(0,len(trainLossList))
plt.plot(xlabels,trainLossList  )

plt.show()









import numpy as np
import TwoLayerNet_BackProp as tln
from dataset.mnist import load_mnist

np.random.seed(1)
(x_train,y_train) , (x_test,y_test) = load_mnist(True,True,True)

iter = 1000
inputSize = 784
hiddenSize = 50
outputSize = 10

trainSize = x_train.shape[0] # Sample Number
batchSize = 100 # Batch Pick Sample Number
lr = 0.1

trainLossList = []
trainAccList = []
testAccList = []

iterPerEpoch = max(trainSize / batchSize , 1)

network = tln.Twolayernet_BackProp( inputSize , hiddenSize , outputSize)

for i in range(iter):
    batchMask = np.random.choice(trainSize , batchSize)
    xBatch = x_train[batchMask]
    yBatch = y_train[batchMask]

    grad = network.BGradient(xBatch , yBatch)

    for key in ('W1','b1','W2','b2'):
        network.params[key] -= lr*grad[key]


    loss = network.Loss(xBatch,yBatch)
    trainLossList.append(loss)

    if i % iterPerEpoch == 0:
        trainAcc = network.Acc(xBatch , yBatch)
        trainAccList.append(trainAcc)
        testAcc = network.Acc(x_test, y_test)
        testAccList.append(testAcc)
        print("Iter : {0} | TrainAcc : {1} | TestAcc : {2}".format(i,trainAcc,testAcc))

    # Perfact !!













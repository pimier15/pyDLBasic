import numpy as np
import os 
import sys
import pickle
from Activation import Activation

from dataset.mnist import load_mnist

sys.path.append(os.pardir)
sys.path.append(os.pardir+"1.ANN")
sys.path.append(os.pardir+"dataset")


class ANNnet:
    def getData(self):
        (x_train,y_train) , (x_test,y_test) = load_mnist(False,True)
        return x_test , y_test

    def initNetwork(self):
        with open (r"F:\00_github\pyDLBasic\vsCode\DL_rawLevel\DL_rawLevel\1.ANN\sample_weight.pkl" , 'rb') as f:
            network = pickle.load(f)
        return network

    def predict(self,network,x):
        W1,W2,W3 = network['W1'],network['W2'],network['W3']
        B1,B2,B3 = network['b1'],network['b2'],network['b3']
        act = Activation()
        a1 = act.sigmoid(np.dot(x ,W1) + B1 )
        a2 = act.sigmoid(np.dot(a1,W2) + B2)
        a3 = act.softMax(np.dot(a2,W3) + B3) 
        return a3


def NotBatch():
    net = ANNnet()
    acc = 0
    x , t = net.getData()
    network = net.initNetwork()
    y = net.predict(network,x)
    for i in range(len(x)):
        
        maxidx = np.argmax(y[i])
        if maxidx == t[i]:
            acc += 1
    
    print("ACcuracy : {0}".format((float(acc)/len(x))))

def Batch(batchsize):
    net = ANNnet()
    acc = 0
    x , t = net.getData()
    network = net.initNetwork()

    for i in range(0,len(x),batchsize):
        x_batch = x[i:i+batchsize]
        y_batch = net.predict(network,x_batch)
        indicies = np.argmax(y_batch , axis= 1)
        acc += np.sum( indicies == t[i:i+batchsize])

    print("ACcuracy : {0}".format((float(acc)/len(x))))



if __name__ == "__main__":
    print("")
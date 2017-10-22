import numpy as np 
from Common import Activation,Diff,SGD,LossFun
from GraphLayer import *  
from collections import OrderedDict


class Twolayernet_BackProp:
    def __init__(self , inputSize , hiddenSize , outputSize , wInitStd = 0.01 , isOneHot = True):
        self.IsOneHot = isOneHot

        self.params = {}
        self.params["W1"] = np.random.randn(inputSize, hiddenSize) * wInitStd
        self.params["b1"] = np.zeros( hiddenSize )
        self.params["W2"] = np.random.randn(hiddenSize, outputSize) * wInitStd
        self.params["b2"] = np.zeros(outputSize)

        # Create Layer
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"] , self.params["b1"]) 
        self.layers["Relu"]  = ReluLayer()
        self.layers["Affine2"] = Affine(self.params["W2"] , self.params["b2"]) 
        self.lastLayer = SofmaxWithLoss(self.IsOneHot)

    def Predict(self,xs):
        for layer in self.layers.values():
            xs = layer.forward(xs)
        return xs

    def Loss(self,xs,ts):
        ys = self.Predict(xs)
        return self.lastLayer.forward(ys , ts )

    def Acc_My(self,xs,ts):
        respredict = self.Predict(xs)
        predict = np.argmax(respredict)
        target = np.argmax(ts)
        
        total = len(predict)
        hitpoint = 0
        for i in range(totla):
            if predict[i] == target[i]:
                hitpoint += 1

        return hitpoint / total * 100 

    def Acc(self,xs,ts):
        ys = self.Predict(xs)
        ys = np.argmax(ys , axis = 1)
        ts = np.argmax(ts , axis = 1)
        acc = np.sum( ys == ts ) / float(xs.shape[0])
        return acc

    def NGradient(self,xs,ts):
        dif = Diff()
        f = lambda W : self.Loss(xs,ts) 
        grad={}
        grad['W1'] = dif.NGradient( f , self.params["W1"] ) 
        grad['b1'] = dif.NGradient( f , self.params["b1"] )
        grad['W2'] = dif.NGradient( f , self.params["W2"] )
        grad['b2'] = dif.NGradient( f , self.params["b2"] )
        return grad

    def BGradient(self,xs,ts):
        self.Loss(xs,ts)

        #BackProp
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad={}
        grad['W1'] = self.layers['Affine1'].dW 
        grad['b1'] = self.layers['Affine1'].db
        grad['W2'] = self.layers['Affine2'].dW
        grad['b2'] = self.layers['Affine2'].db
        return grad

if __name__ == "__main__":
    net = Twolayernet(784,100,10)

    x = np.random.randn(100,784)
    y = np.random.randn(100,10)
    
    res = net.Predict(x)
    print(res)








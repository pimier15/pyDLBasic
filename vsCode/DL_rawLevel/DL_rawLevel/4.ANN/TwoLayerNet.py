import numpy as np 
from Common import Activation,Diff,SGD,LossFun



class Twolayernet:
    def __init__(self , inputSize , hiddenSize , outputSize , wInitStd = 0.01):

        self.params = {}
        self.params["W1"] = np.random.randn(inputSize, hiddenSize) * wInitStd
        self.params["b1"] = np.zeros( hiddenSize )
        self.params["W2"] = np.random.randn(hiddenSize, outputSize) * wInitStd
        self.params["b2"] = np.zeros(outputSize)

    def Predict(self,xs):
        W1,W2 = self.parmas["W1"] , self.params["W2"]
        b1,b2 = self.parmas["b1"] , self.params["b2"]
        a1 =  np.dot(xs,W1) + b1
        z1 = Activation().sigmoid(z1)
        a2 = np.dot(a1,W2)+b2
        z2 = Activation().softmax(z2)
        return a2

    def Loss(self,xs,ts):
        respredict = self.Predict(xs)
        lsFun = LossFun()
        ceeLoss = lsFun.CEE(ts,respredict)
        return ceeLoss

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
        y = self.Predict(xs)
        y = np.argmax(y , axis = 1)
        t = np.argmax(ts , axis = 1)
        
        acc = np.sum( y == t ) / float(x.shape[0])
        return acc

    def NGradient(self,xs,ts):
        dif = Diff()

        y = self.Predict(xs)
        f = lambda W : self.Loss(xs,ts) 

        #grad={}
        #grad['W1'] = dif.NGradient() 
        #grad['b1'] = 
        #grad['W2'] = 
        #grad['b2'] = 


if __name__ == "__main__":
    net = Twolayernet(784,100,10)

    x = np.random.randn(100,784)
    y = np.random.randn(100,10)
    
    res = net.Predict(x)
    print(res)







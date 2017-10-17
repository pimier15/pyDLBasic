import numpy as np
from Common import Activation , Diff , SGD ,LossFun

Act = Activation()
Dif = Diff()
Sgd = SGD()
Loss = LossFun()

class simpleANN:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self,x):
       
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y = Act.softMax(z)
        return Loss.CEE(t , y)

    def train(self,x,t):
        self.W =  Sgd.sgd(self.predict , self.W )

if __name__ == '__main__':
    # find  x0,x1 => argmin f1 
    net = simpleANN()
    print(net.W)

    x = np.array([0.6,0.9])
    t = np.array([])
    print(net.predict(x))

    net.train(x , )

        
        


import numpy as np
from Common import LossFun , Activation

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward( self , dout ):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x+y
        return out

    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy


class DivideLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forawrd(self,x,y):
        if y != 0 :
            self.x = x
            self.y = y
            out = x/y
            return out
        return None 

    def backward(self,dout):
        dx = self.y**-1
        dy = -self.x * self.y**-2
        return dx,dy
        
class NSquare:
    def __init__(self):
        self.x = None
        self.n  = None

    def forward(self,x,n):
        self.x = x
        self.n = n
        return x**n

    def backward(self,dout):
        dx = dout*self.n*self.x**(self.n-1)
        return dx



class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = ( x <= 0 )
        out = x.copy()
        out[self.mask] = 0
        return out 

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SigmoidLayer:
    def __init__(self):
        self.out = None

    def foward(self,x):
        self.out = 1/(1 + np.exp(-x))
        return self.out

    def backward(self,dout):
        dx = self.out*(1.0 - self.out)*dout
        return dx

class Affine:
    def __init__(self , w,  b):
        self.W = w
        self.B = b
        self.x = None
        self.dW = None 
        self.db = None # why save? => use this for update parameter when doing training

    def forward( self , x ):
        self.x = x
        out = np.dot(x,self.W) + self.B
        return out

    def backward(self , dout):
        dx = np.dot(dout , self.W.T)
        self.dW = np.dot(self.x.T , dout ) 
        self.db = np.sum(dout , axis = 0)
        return dx

class SofmaxWithLoss:
    def __init__(self , IsOneHot):
        self.loss = None
        self.y = None
        self.t = None
        self.isOneHot = IsOneHot

    def forward(self,xs,ts):
        self.t = ts
        self.y = Activation().softmax(xs) 
        self.loss = LossFun().CEE(self.t,self.y)
        return self.loss 

    def backward(self , dout = 1):
        batchSize = self.t.shape[0]
        dx = (self.y - self.t) / batchSize
        return dx






        



if __name__ == "__main__":
    x
    def AppleExample():
        ApNum = 2
        ApVal = 100
        Tax = 1.1

        K1 = Mulayer()
        K2 = Mulayer()

        K1out = K1.forward(ApNum , ApVal)
        K2out = K2.forward(K1out , Tax)

        print(K1out)
        print(K2out)
        print(K1.x)
        print(K1.y)
        print('-'*8)
        print(K2.x)
        print(K2.y)
        print('='*8)
        print(" Calc Partial derivitive")
        
        dK2out = 1
        dK1 , dTax = K2.backward(dK2out)
        print(" dTax : %s" % dTax)
        
        dApNum , dApVal = K1.backward(dK1)
        print(" dApNum : {0} , dApVal : {1}".format(dApNum , dApVal))

    def AppleAndOrangeExample():
        ApNum = 2 
        ApVal = 100
        OrNum = 3
        OrVal = 150
        Tax   = 1.1

        L1 = MulLayer()
        L2 = MulLayer()
        L3 = AddLayer()
        L4 = MulLayer()

        outL1 = L1.forward( ApNum , ApVal )
        outL2 = L2.forward( OrNum , OrVal )
        outL3 = L3.forward( outL1 , outL2 )
        outL4 = L4.forward( outL3 , Tax )

        dL4 = 1
        dL3 , dTax = L4.backward( dL4 )
        dL1 , dL2  = L3.backward( dL3 )
        dApNum , dApVal = L1.backward( dL1 )
        dOrNum , dOrVal = L2.backward( dL2 )

        print()


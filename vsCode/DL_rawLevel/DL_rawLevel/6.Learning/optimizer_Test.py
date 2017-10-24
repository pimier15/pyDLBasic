import numpy as np
import matplotlib.pyplot  as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import *
from GraphLayer import *

class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr

    def update(self, params , grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]

class Momentum:
    def __init__(self,lr = 0.01 , momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grad):
        if self.v is None:
            self.v = {}
            for key , val in parmas.items():
                self.v[key] = np.zeros_like(val)

            for key in params.keys():
                self.v[key] = self.momentum*self.v[key] - self.lr*grad[key]
                params[key] += self.v[key]




def CreateMap(x,y):
    z = 1.0/20.0 *(x**2) + y**2
    return z

def CreateMap2(x,y):
    z = x*1000 + y
    return z


x = np.arange(-10,10,0.1)
y = np.arange(-10,10,0.1)
xs ,ys = np.meshgrid(x,y)
z = CreateMap(xs,ys)

fig = plt.figure()
#ax =fig.gca(projection='3d')
plt.hold(True)




a = -7
b = 2
Params = {}
Params['a'] = a
Params['b'] = b

Grad = {}
Grad['a'] = 10000
Grad['b'] = 10000

iter = 10

L1 = MulLayer()
L2 = MulLayer()
L3 = MulLayer()
L4 = AddLayer()

scaterAList = []
scaterBList = []
scaterZList = []


for i in range(1000):
    outL1 = L1.forward(Params['a'],Params['a'])
    outL2 = L2.forward(outL1,0.05)
    outL3 = L3.forward(Params['b'],Params['b'])
    outL4 = L4.forward(outL2 , outL3)
    
    backL4 = 1
    Grad['b'] , _ = L3.backward(1)
    Grad['a'] , _ = L1.backward( 0.05  )

    

    if i % 100 == 0:
        scaterAList.append(Params['a'])
        scaterBList.append(Params['b'])
        scaterZList.append(outL4+10)

        #print("L : " , outL4)
        #print("Real : " , CreateMap(Params['a'],Params['b']))
        #print()
    SGD().update(Params,Grad)

plt.scatter(xs,ys,c = z )


#sct = ax.scatter(scaterAList,scaterBList,scaterZList, s = 100 , c = 'black')
#surf = ax.plot_surface(xs,ys,z, cmap = cm.jet )

plt.show()









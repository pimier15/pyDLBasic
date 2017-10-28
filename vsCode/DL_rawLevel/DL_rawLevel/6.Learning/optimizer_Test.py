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

class AdaGrad:
	def __init__(self , lr = 0.01):
		self.lr = lr
		self.h = None

	def update(self , params , grads):
		if self.h == None:
			self.h = {}
			for key , val in params.items():
				self.h[key] = np.zeros_like(val)
		for key , val in params.keys():
			self.h[key] += grads[key]*grads[key]
			params[key] -= self.lr*grads[key] / (np.sqrt(self.h[key]) + 1e-7)


	


def CreateMap(x,y):
    z = 1.0/20.0 *(x**2) + y**2
    return z

def CreateMap2(x,y):
    z = x*1000 + y
    return z


x = np.arange(-10,10,0.5)
y = np.arange(-10,10,0.5)
xs ,ys = np.meshgrid(x,y)
z = CreateMap(xs,ys)

fig = plt.figure()
ax =fig.gca(projection='3d')
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


sct = ax.scatter(scaterAList,scaterBList,scaterZList, s = 100 , c = 'black')
#surf = ax.plot_surface(xs,ys,z, cmap = cm.jet )

plt.show()









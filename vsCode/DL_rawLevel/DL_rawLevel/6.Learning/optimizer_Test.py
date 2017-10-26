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


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)




def CreateMap(x,y):
    z = 1.0/20.0 *(x**2) + y**2
    return z

def df (x,y):
    dx = 0.1*x
    dy  = 2.0*y
    return dx,dy

x = np.arange(-10,10,0.1)
y = np.arange(-5,5,0.1)
xs ,ys = np.meshgrid(x,y)
z = CreateMap(xs,ys)

fig = plt.figure()
#ax =fig.gca(projection='3d')


a = -7.0
b = 2.0
Params = {}
Params['a'] = a
Params['b'] = b

Params1 = {}
Params1['a'] = a
Params1['b'] = b

Grad = {}
Grad['a'] = 0
Grad['b'] = 0

Grad1 = {}
Grad1['a'] = 0
Grad1['b'] = 0

iter = 10

L1 = NSquare()
L2 = MulLayer()
L3 = NSquare()
L4 = AddLayer()

scaterAList = []
scaterBList = []
xl = []
yl = []


for i in range(30):
    outL1 = L1.forward(Params['a'],2)
    outL2 = L2.forward(outL1,0.05)
    outL3 = L3.forward(Params['b'],2)
    outL4 = L4.forward(outL2 , outL3)

    dx,dy = df(Params1['a'],Params1['b'])
    backL4 = 1
    Grad['a'] = L1.backward(0.05 )
    Grad1['a']  = dx
    Grad['b'] = L3.backward(1.0)
    Grad1['b'] = dy
    


   

    print("Iter : {0} , Back : {1} , dx : {2}".format(i, Grad['a'] , Grad1['a']))
    
    scaterAList.append(Params['a'])
    scaterBList.append(Params['b'])

    xl.append(Params['a'])
    yl.append(Params['b'])

    Adagrad(lr=0.3).update(Params,Grad)
  



plt.xlim(-10,10)
plt.ylim(-10,10)

plt.plot(scaterAList,scaterBList, c = 'red')
plt.plot(xl,yl, c = 'b')
plt.contour(xs,ys,z)

plt.show()









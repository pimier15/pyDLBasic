from Common import Diff
from TwoLayerNet_BackProp import *
from Optimizer import *
from matplotlib import pyplot as plt


inputSize = 784
hiddenSize = 50
outputSize = 10

def TestFunc(x,y):
    z = 1.0/20.0 *(x**2) + y**2
    return z

def df(x,y):
    dx = 0.1*x
    dy = 2*y
    return dx , dy

init_pos = (-7.0 , 2.0)
params = {}
params['x'] , params['y'] = init_pos[0] , init_pos[1]
grad = {}
grad['x'] , grad['y'] = 0,0

optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr = 0.95)
optimizers['Momentum'] = Momentum(lr = 0.1)
optimizers['Adam'] = Adam(lr = 0.3)

idx =1

for key in optimizers:
    optimizer = optimizers[key]
    xLog = []
    yLog = []

    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):
        xLog.append(params['x'])
        yLog.append(params['y'])

        grad['x'] , grad['y'] = df(params['x'],params['y'])

        print(grad['x'])
        optimizer.update(params,grad)

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y) 
    Z = TestFunc(X, Y)

 

    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(xLog, yLog, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
plt.show()







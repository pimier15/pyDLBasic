import numpy as np 
import Diff as Df
import matplotlib.pyplot as plt

class SGD:
    def gradient(self,f,x):
        h = 1e-4
        grad = np.zeros_like(x)

        for idx in range(x.size):
            tmpVal = x[idx]
            
            x[idx] = tmpVal + h
            fxh1 = f(x)

            x[idx] = tmpVal - h
            fxh2 = f(x)

            grad[idx] = (fxh1 - fxh2) / (2*h)
            x[idx] = tmpVal
        return grad

    def sgd(self,f,x0 , lr = 0.01 , iter = 1000 ):
        x = x0
        for i in range(iter):
            grad = self.gradient(f , x)
            x -= lr * grad
        return x

    def sgdHistory(self,f,x0 , lr = 0.01 , iter = 100 ):
        his = []
        his.append(x0)
        x = x0
        for i in range(iter):
            grad = self.gradient(f , x)
            x -= lr * grad
            his.append(list(x))
        return x , his


if __name__ == '__main__':
    # find  x0,x1 => argmin f1 

    def f1(x):
        return x[0]**2 + x[1]**2

    init_x = np.array([-3.0,4.0])
    sgd =SGD()

    res1 , his1 = sgd.sgdHistory(f1,init_x, lr = 0.001)
    res2 , his2 = sgd.sgdHistory(f1,init_x,lr = 0.3)
    plt.xlabel('x0')
    plt.ylabel('x1')
    #plt.xlim(0,6)
    #plt.ylim(0,6)
    plt.scatter([x[0] for x in his1] , [x[1] for x in his1] ,s = 10 , c = 'b' , marker = '*')
    plt.scatter([x[0] for x in his2] , [x[1] for x in his2] ,s = 10 , c = 'r' , marker = 'o' )
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

class Diff:
    def diff1(self,f,x):
        h = 1e-4
        return (f(x+h)-f(x-h)) / (2*h)

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

    def diff_Line(self,a,x0,y0,x):
        return a*(x - x0)+y0


if __name__ == '__main__':
    df = Diff()
    def f(x):
        return 0.1*x**2 + 3*x + 9.0

    def f2(x):
        return x[0]**2 + x[1]**2

    ## 1D Diffrential
    x = np.arange(-10,10,0.1)
    y = f(x)

    x0 = x[50]
    y0 = y[50]

    diffline = df.diff_Line( df.diff1(f,x0) , x0,y0, x )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x,y,'r')
    plt.plot(x0,y0,'g+')
    plt.plot(x,diffline,'b')
    #plt.show()

    ## Partial 
    def ftmp1(x0):
        return x0**2 + 4.0**2.0

    res = df.diff1(ftmp1 , 3 )
    print(res)

    grad1 = df.gradient(f2 , np.array([3.0,4.0]))
    grad2 = df.gradient(f2 , np.array([0.0,4.0]))
    print(grad1)
    print(grad2)




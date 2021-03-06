import numpy as np

class Activation:
    def step_function(self,x):
        y = x > 0
        return y.astype(np.int)

    #왜 sigmoid? 
    # 출력이 0~1 사이이고, 미분이 쉬운 함수들중 가장 그럴듯 해서 많이 사용하게됨. 미분은 exp이 쉽기때문에
    def sigmoid(self,x):  
        return 1/(1 + np.exp(-x))

    def RelU(self,x):
        return np.maximum(0,x)

     #왜 softmax? 
    # 출력이 0~1 사이이고, 미분이 쉬운 함수들중 가장 그럴듯 해서 많이 사용하게됨. 미분은 exp이 쉽기때문에
    def softmax(self,xs):
        if xs.ndim == 2:
            xs = xs.T
            xs = xs - np.max(xs, axis=0)
            y = np.exp(xs) / np.sum(np.exp(xs), axis=0)
            return y.T 

        xs = xs - np.max(xs) # 오버플로 대책
        return np.exp(xs) / np.sum(np.exp(xs))

class LossFun:
    def MSE(self,ts,ys):
        batch_size = ys.shape[0]
        if ys.ndim == 1:   
            ts = ts.reshape(1,ts.size)
            ys = ys.reshape(1,ys.size)
        return 1/2*(np.sum((ys - ts)**2))  / batch_size

    #def CEE(self,ts,ys,IsOneHot = True):
    #    batch_size = ys.shape[0]
    #    if ys.ndim == 1:   
    #        ts = ts.reshape(1,ts.size)
    #        ys = ys.reshape(1,ys.size)
    #   
    #    if IsOneHot :
    #         return -np.sum(ts*np.log(ys + 1e-7) ) / batch_size
    #    else: 
    #         return -np.sum(np.log(ys[np.arange(batch_size) , ts] + 1e-7) ) / batch_size

    def CEE(self,ts,ys):
        batch_size = ys.shape[0]

        if ys.ndim == 1:   
            ts = ts.reshape(1,ts.size)
            ys = ys.reshape(1,ys.size)

        if ts.size == ys.size:
            ts = ts.argmax(axis=1)
       
        return -np.sum(np.log(ys[np.arange(batch_size), ts])) / batch_size
    

        

class Diff:
    def NGradient(self,f,x):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)
        
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x) # f(x+h)
            
            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            
            x[idx] = tmp_val # 값 복원
            it.iternext()   
            
        return grad

    def diff(self,f,x):
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


class SGD:
    def NGradient(self,f,s):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)
        
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x) # f(x+h)
            
            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            
            x[idx] = tmp_val # 값 복원
            it.iternext()   
            
        return grad
        


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



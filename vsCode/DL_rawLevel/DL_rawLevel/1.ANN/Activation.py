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
    def softMax(self,xs):
        mx = np.max(xs)
        expx = np.exp( xs - mx )
        expsum = np.sum(expx)
        y = expx / expsum
        return y

if __name__ == "__main__":
    a = [ 1 ,3]
    print(Activation().softMax(a))


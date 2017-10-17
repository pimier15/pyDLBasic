import numpy as np
import os
import sys
from Activation import Activation


sys.path.append(os.pardir)
sys.path.append(os.pardir+"dataset")

from dataset.mnist import load_mnist

(x_train,y_train) , (x_test,y_test) = load_mnist(True,True)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 랜덤하게 뽑아오기
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)

x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]

x = [1,2,3]
x2 = [10,100 ,1000]

xa = np.array(x)
x1a = np.array(x2)

print(xa*x1a)
print( np.multiply( xa,x1a))
print( np.dot( xa,x1a))

print()




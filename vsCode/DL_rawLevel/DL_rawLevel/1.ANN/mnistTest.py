import numpy as np
import os
import sys
from Activation import Activation


sys.path.append(os.pardir)
sys.path.append(os.pardir+"dataset")

from dataset.mnist import load_mnist

(x_train,y_train) , (x_test,y_test) = load_mnist(False,True)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)







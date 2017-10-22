import numpy as np
import TwoLayerNet_BackProp as tln
from dataset.mnist import load_mnist

np.random.seed(1)
(x_train,y_train) , (x_test,y_test) = load_mnist(True,True,True)

x_batch = x_train[:3]
y_batch = y_train[:3]

inputSize = 784
hiddenSize = 50
outputSize = 10

network = tln.Twolayernet_BackProp( inputSize , hiddenSize , outputSize)
print('Gradient is Started')
nGradient = network.NGradient(x_batch , y_batch)
bGradient = network.BGradient(x_batch , y_batch)

print('Gradient is Done')
for key in nGradient.keys():
    diff = np.average( np.abs( bGradient[key] - nGradient[key] ) )
    print("key {0} : Error : {1}".format(key,diff))





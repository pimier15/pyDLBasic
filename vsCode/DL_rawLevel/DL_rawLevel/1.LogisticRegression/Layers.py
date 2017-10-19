import numpy as np
import Activation

act = Activation.Activation()

input = np.array([1.0 , 0.5])
layer1 = np.array([ [0.1 , 0.2 , 0.3] , [0.1 , 0.3 , 0.5] ])
bias1 = np.array([0.1,0.2,0.3])
A1 = np.dot(input,layer1) + bias1
z1 = act.sigmoid(A1)

layer2 = np.array( 
    [
        [0.1,0.2],
        [0.2,0.2],
        [0.7,0.3]
    ])
bias2 = np.array([1.0 , 1.0])
A2 = np.dot(z1,layer2) + bias2
z2 = act.sigmoid(A2)

layer3 = np.array([[0.4,0.5],
                    [0.1,0.2]])
bias3 = np.array([1,1])

A3 = np.dot(z2,layer3) + bias3

Y = act.sigmoid( A3 )

class Net:
    def init_net(self):
        network={}
        network['W1'] = np.array([ [0.1 , 0.2 , 0.3] , [0.1 , 0.3 , 0.5] ]) 
        network['B1'] = np.array([0.1,0.2,0.3])
        network['W2'] = np.array( [[0.1,0.2],
                                   [0.2,0.2],
                                   [0.7,0.3]])
        network['B2'] = np.array([1.0 , 1.0]) 
        network['W3'] = np.array([[0.4,0.5],
                                [0.1,0.2]]) 
        network['B3'] = np.array([1,1])
        return network

    def forward(self,network,x):
        W1,W2,W3 = network['W1'],network['W2'],network['W3']
        B1,B2,B3 = network['B1'],network['B2'],network['B3']

        act = Activation.Activation()

        a1 = act.sigmoid(np.dot(x ,W1) + B1 )
        a2 = act.sigmoid(np.dot(a1,W2) + B2)
        a3 = act.sigmoid(np.dot(a2,W3) + B3) 

        return a3

net = Net()

print(Y)
print(net.forward(net.init_net(),input))

        



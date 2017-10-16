import Activation
import matplotlib.pyplot as plt
import numpy as np

actv = Activation.Activation()
list = [ x / 1000.0 for x in range(-9000,9000)]

step_res = actv.step_function(np.asarray(list))
sigmoid_res = actv.sigmoid(np.asarray(list))

plt.plot(list , sigmoid_res)
plt.show()








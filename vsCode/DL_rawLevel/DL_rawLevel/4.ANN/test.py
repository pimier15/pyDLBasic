import numpy as np


ys = np.array([[1,2] , [3,4]])
#ys = np.array([1,2 , 3,4])
ts = np.array([[0,1] , [0,1]])
#ts = np.array([0,1 , 0,1])
IsOneHot = True
res = None

batch_size = ys.shape[0]
#res = -np.sum(ts*np.log(ys + 1e-7) ) / batch_size
#res = -np.sum(np.log(ys[np.arange(batch_size) , t]) ) / batch_size

ly = np.log(ys + 1e-7)
lymulti = ts * ly 

print(ly)
print(lymulti)
print()

res1 = np.sum(lymulti , axis = 0) 
res2  =np.sum(lymulti , axis = 1) 
res2  =np.sum(lymulti ) 


print()



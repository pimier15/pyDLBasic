import numpy as  np

x = np.array([[1 , -1] ,[1,2],[3,4]])
y = x.T

mask = (x < 0)
print(mask)

x[mask] = 0
print(x)



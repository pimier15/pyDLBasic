import numpy as np

class LossFun:
    def MSE(self,ts,ys):
        batch_size = ys.shape[0]
        if ys.ndim == 1:   
            ts = ts.reshape(1,ts.size)
            ys = ys.reshape(1,ys.size)
        return 1/2*(np.sum((ys - ts)**2))  / batch_size

    def CEE(self,ts,ys):
        batch_size = ys.shape[0]
        if ys.ndim == 1:   
            ts = ts.reshape(1,ts.size)
            ys = ys.reshape(1,ys.size)
        return -np.sum(ts*np.log(ys[np.arange(batch_size) , ts] + 1e-7) ) / batch_size







if __name__ == "__main__":
    y =  [[0.1 , 0.05 , 2.1 ] , [0.1 , 0.05 , 2.1 ] , [0.1 , 0.05 , 2.1 ] ]
    test = [[0,0,1],[0,1,0],[1,0,0]]
    test1 = [[3,2,1]]
    t1 = np.array(test)
    t2 = np.array(test1)                                                     

    #하고 싶은건  e1*log(0.1) + e2*log(0.05) + e3*log(2.1)
    # test1 은 [3,2,1]이다 어떻게?
    
    onehot = y[test1[0] ]







    
    loss = LossFun()
    y =  [[0.1 , 0.05 , 2.1 , 2.7] , [0.1 , 0.05 , 2.1 , 2.7] , [0.1 , 0.05 , 2.1 , 2.7] , [0.1 , 0.05 , 2.1 , 2.7]]
    t1 = [[0 , 1 , 0 , 0],[0 , 1 , 0 , 0],[0 , 1 , 0 , 0],[0 , 1 , 0 , 0]]
    t2 = [[0 , 0 , 1 , 0],[0 , 0 , 1 , 0],[0 , 0 , 1 , 0],[0 , 0 , 1 , 0]]

    yd1 = np.array([0.1 , 0.05 , 2.1 , 2.7])
    td1 = np.array([0,1,0,0]               )
    print( loss.CEE(td1,yd1))

    print( "MSE t1 " + str(loss.MSE(np.array(t1) , np.array(y))) )
    print( "MSE t2 " + str(loss.MSE(np.array(t2) , np.array(y))) )
    print( "CEE t1 " + str(loss.CEE(np.array(t1) , np.array(y))) )
    print( "CEE t2 " + str(loss.CEE(np.array(t2) , np.array(y))) )








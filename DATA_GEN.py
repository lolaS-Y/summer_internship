from z3 import *
import numpy as np


def data_gen():
    np.random.seed(1) 
    A = np.random.randint(low = -25, high=26, size=(8000,2))
    B = np.zeros(((A.shape[0],1)))
    for i in range(A.shape[0]):
        x1 = A[i,0]
        x2 = A[i,1]
        if (((-3*x1+6*x2-2>=0) and (x1+x2-5 >= 0)) or ((x1+7*x2-4 <=0) and (x1-2*x2+4 > 0))):
            B[i,0] = 1
    return A, B

import unittest
import numpy as np
from cul_func import Omega,S,Matrix,Acc_n,Kk
import csv
import pandas as pd

data = pd.read_csv("data/1.csv",sep=";")
first = data.loc[0,:]
second = data.loc[1,:]
third = data.loc[2,:]

delta_t = 1/120


class Testpbl(unittest.TestCase):
    data = [0,-4.773060,-1.769106,8.427839,-0.030030,0.022133,0.002239]
    npdata = np.array(data)

    def test_Cul_Omega(self):
        result = Omega(first)
        ans = [[0,-0.002239,0.022133],[0.002239,0,0.030030],[-0.022133,-0.030030,0]]
        np.testing.assert_array_equal(result,ans)

    # def test_Cul_S(self):
    #     a = [-0.030030,0.022133,0.002239]
    #     a = np.array(a)
    #     result = S(a)
    #     ans = [[0,-0.002239,0.022133],[0.002239,0,0.030030],[-0.022133,-0.030030,0]]
    #     np.testing.assert_array_equal(result,ans)

    def test_Cul_matrix(self):
        data = [0,-4.773060,-1.769106,8.427839,-0.030030,0.022133,0.002239]
        Omegak = Omega(first)
        delta_t: float = 1/120
        result = Matrix(2 * np.identity(3,dtype=float) + Omegak * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak*delta_t))
        ans = [[1,-1.86813*(10**-5),1.84442*(10**-4)],[1.86572*(10**-5),1,2.5024*(10**-4)],[-1.84444*(10**-4),-2.5025*(10**-4),1]]
        np.testing.assert_array_almost_equal(result,ans)

    def test_Cul_Kk(self):
        r = [0.01**2,0.01**2,0.01**2]
        R = np.diag(r)
        H = np.hstack([np.hstack([np.zeros((3,3)),np.zeros((3,3))]),np.identity(3,dtype=np.float64)])
        Pk = matrix = np.array([
            [1.3889e-08, 0, 0, 0, 0, 0, 0, -5.7119e-10, 1.6636e-12],
            [0, 1.3889e-08, 0, 0, 0, 0, 5.7119e-10, 0, -1.1985e-12],
            [0, 0, 1.3889e-08, 0, 0, 0, -1.6636e-12, 1.1985e-12, 0],
            [0, 0, 0, 4.8226e-13, 0, 0, 0, 5.7873e-11, 0],
            [0, 0, 0, 0, 4.8226e-13, 0, 0, 0, 5.7873e-11],
            [0, 0, 0, 0, 0, 4.8226e-13, 0, 5.7873e-11, 0],
            [0, 5.7119e-10, -1.6636e-12, 5.7873e-11, 0, 0, 1.3936e-08, -2.8707e-16, -9.8561e-14],
            [-5.7119e-10, 0, 1.1985e-12, 0, 5.7873e-11, 0, -2.8707e-16, 1.3936e-08, -1.3682e-13],
            [1.6636e-12, -1.1985e-12, 0, 0, 0 , 5.7873e-11, -9.8561e-14, -1.3682e-13, 1.3889e-08],
        ], dtype=np.float64)
        result = Kk(H,Pk,R)
        ans = np.array([
            [-1.9142e-33, -5.7111e-06,  1.6634e-08],
            [ 5.7111e-06,  3.0815e-33, -1.1983e-08],
            [-1.6634e-08,  1.1983e-08,  5.5536e-33],
            [ 5.7865e-07,  1.6609e-18,  5.7025e-16],
            [ 1.6609e-18,  5.7865e-07,  7.9157e-16],
            [ 5.7025e-16,  7.9157e-16,  5.7865e-07],
            [ 1.3934e-04, -2.8699e-12, -9.8534e-10],
            [-2.8699e-12,  1.3934e-04, -1.3678e-09],
            [-9.8534e-10, -1.3678e-09,  1.3887e-04],
        ])
        np.testing.assert_array_almost_equal(result,ans)
        
    # def test_Cul_acc_n(self):
    #     Omegak_1 = Omega(second)
    #     Ck_1 = Matrix(2 * np.identity(3,dtype=float) + Omegak_1 * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak_1*delta_t))
    #     Omegak = Omega(third)
    #     Ck = Matrix(2 * np.identity(3,dtype=float) + Omegak * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak*delta_t))
    #     result = Acc_n(Ck,Ck_1,second)   
    #     ans = [[-4.763233],[-1.678572],[8.581781]]
    #     np.testing.assert_array_almost_equal(result,ans,5)
        



if __name__ == '__main__':
    unittest.main()
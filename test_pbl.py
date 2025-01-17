import unittest
import numpy as np
from cul_func import Omega,S,Matrix,Acc_n
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

    def test_Cul_S(self):
        a = [-0.030030,0.022133,0.002239]
        a = np.array(a)
        result = S(a)
        ans = [[0,-0.002239,0.022133],[0.002239,0,0.030030],[-0.022133,-0.030030,0]]
        np.testing.assert_array_equal(result,ans)

    def test_Cul_matrix(self):
        data = [0,-4.773060,-1.769106,8.427839,-0.030030,0.022133,0.002239]
        Omegak = Omega(first)
        delta_t: float = 1/120
        result = Matrix(2 * np.identity(3,dtype=float) + Omegak * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak*delta_t))
        ans = [[1,-1.86813*(10**-5),1.84442*(10**-4)],[1.86572*(10**-5),1,2.5024*(10**-4)],[-1.84444*(10**-4),-2.5025*(10**-4),1]]
        np.testing.assert_array_almost_equal(result,ans)

    def test_Cul_F(self):
        
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
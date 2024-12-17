import unittest
import numpy as np
from pbl import Cul_Omega,Cul_S,Cul_matrix,Cul_acc_n
import csv

csv_data = []
delta_t: float = 1/120

class Testpbl(unittest.TestCase):
    data = [0,-4.773060,-1.769106,8.427839,-0.030030,0.022133,0.002239]
    npdata = np.array(data)

    def test_Cul_Omega(self):
        a = [0.008333,-4.773060,-1.769106,8.427839,-0.030030,0.022133,0.002239]
        result = Cul_Omega(a)
        ans = [[0,-0.002239,0.022133],[0.002239,0,0.030030],[-0.022133,-0.030030,0]]
        np.testing.assert_array_equal(result,ans)

    def test_Cul_S(self):
        a = [-0.030030,0.022133,0.002239]
        a = np.array(a)
        result = Cul_S(a)
        ans = [[0,-0.002239,0.022133],[0.002239,0,0.030030],[-0.022133,-0.030030,0]]
        np.testing.assert_array_equal(result,ans)

    def test_Cul_matrix(self):
        data = [0,-4.773060,-1.769106,8.427839,-0.030030,0.022133,0.002239]
        Omegak = Cul_Omega(data)
        delta_t: float = 1/120
        result = Cul_matrix(2 * np.identity(3,dtype=float) + Omegak * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak*delta_t))
        ans = [[1,-1.86813*(10**-5),1.84442*(10**-4)],[1.86572*(10**-5),1,2.5024*(10**-4)],[-1.84444*(10**-4),-2.5025*(10**-4),1]]
        np.testing.assert_array_almost_equal(result,ans)

    def test_Cul_acc_n(self):
        Omegak_1 = Cul_Omega(csv_data[1])
        Ck_1 = Cul_matrix(2 * np.identity(3,dtype=float) + Omegak_1 * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak_1*delta_t))
        Omegak = Cul_Omega(csv_data[2])
        Ck = Cul_matrix(2 * np.identity(3,dtype=float) + Omegak * delta_t , np.linalg.inv(2 * np.identity(3,dtype=float) - Omegak*delta_t))
        result = Cul_acc_n(Ck,Ck_1,csv_data[1])   
        ans = [[-4.763233],[-1.678572],[8.581781]]
        np.testing.assert_array_almost_equal(result,ans,5)
        



if __name__ == '__main__':
    with open("data/1.csv") as f:
        reader = csv.reader(f,delimiter =';')
        i = 0
        for row in reader:
            csv_data.append(row)
            i = i+1
    unittest.main()
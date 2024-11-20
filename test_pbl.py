import unittest
import numpy as np
from pbl import Cul_Omega,Cul_S,Cul_matrix

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

    def test_Cul_matrix():
        data = [0,-4.773060,-1.769106,8.427839,-0.030030,0.022133,0.002239]
        Omegak = Cul_Omega(data)
        result = Cul_matrix(Omegak,Omegak)
        ans = [[-0.00049488281,-0.00066465399,-0.00006723717],[-0.00066465399,-0.000906814021,0.000049555787],[-0.00006723717,0.000049555787,-0.001]]
        np.testing.assert_array_almost_equal(result,ans)


if __name__ == '__main__':
    unittest.main()
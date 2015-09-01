import unittest

import numpy as np
from scipy import linalg

from givens import givensise

class LinearModelOperationsTestCase(unittest.TestCase):
    def setUp(self):
        
        # Test parameters
        self.dMax = 10


class GivensiseTestCase(LinearModelOperationsTestCase):
    def runTest(self):
        
        for d in range(1,self.dMax+1):
            for r in range(1,d):
                
                # Random matrix of orthogonal columns
                X = np.random.randn(d,d)
                A = np.dot(X,X.T)
                eVal,eVec = linalg.eigh(A)
                U = eVec[:,:2]
                
                # Factorise it
                Uc,E,Ur = givensise(U)
                
                # Reconstruct
                shouldBeU = np.dot(Uc,np.dot(E,Ur))
                
                # Test
                np.testing.assert_almost_equal(U,shouldBeU)                
import unittest

import numpy as np

from kalman import GaussianDensity
from linear_models import BasicLinearModel

class LinearModelOperationsTestCase(unittest.TestCase):
    def setUp(self):
        
        # Basic parameters
        self.K = 10
        self.ds = 2
        self.do = 1
        
        # System matrices
        params = dict()
        params['F'] = np.array([[0.9,0.81],[0,0.9]])
        params['Q'] = np.array([[1,0],[0,1]])
        params['H'] = np.array([[1,0]])
        params['R'] = np.array([[1]])
        
        # Create model
        prior = GaussianDensity(np.array([0,0]), np.array([[100,0],[0,100]]))
        self.model = BasicLinearModel(self.ds, self.do, prior, params)


class SimulateDataTestCase(LinearModelOperationsTestCase):
    def runTest(self):
        true_state = np.array([ [ 17.64052346,   4.00157208],
                                [ 20.09648249,   5.84230807],
                                [ 24.68666177,   4.28079939],
                                [ 26.63553151,   3.70136224],
                                [ 26.86686292,   3.74182452],
                                [ 27.35509806,   4.82191557],
                                [ 29.2863776 ,   4.46139903],
                                [ 30.41533628,   4.34893346],
                                [ 32.39051783,   3.70888185],
                                [ 32.46872804,   2.48389792] ])
        
        true_observ = np.array([[ 15.08753364],
                                [ 20.75010108],
                                [ 25.55109797],
                                [ 25.89336649],
                                [ 29.13661755],
                                [ 25.90073239],
                                [ 29.33213611],
                                [ 30.22815243],
                                [ 33.92329704],
                                [ 33.93808681]])
        
        np.random.seed(0)
        state, observ = self.model.simulate_data(self.K)
        
        np.testing.assert_almost_equal(state,true_state)
        np.testing.assert_almost_equal(observ,true_observ)


class FilterTestCase(LinearModelOperationsTestCase):
    def runTest(self):
        true_prd_mn = np.array([[  0.        ,   0.        ],
       [ 13.44433691,   0.        ],
       [ 24.88486417,   7.00653871],
       [ 28.88907476,   6.64060538],
       [ 28.20304549,   4.84335329],
       [ 30.25419611,   4.68487149],
       [ 26.64137582,   2.72030464],
       [ 28.88527605,   3.37102832],
       [ 30.0768482 ,   3.49437001],
       [ 33.7651837 ,   4.46380605]])
        
        true_prd_vr = np.array([[[ 100.        ,    0.        ],
        [   0.        ,  100.        ]],
       [[  67.4119802 ,   72.9       ],
        [  72.9       ,   82.        ]],
       [[   6.18455567,    4.01063471],
        [   4.01063471,    4.49721824]],
       [[   3.99286853,    2.09851187],
        [   2.09851187,    2.82927283]],
       [[   3.53816888,    1.76000025],
        [   1.76000025,    2.57728417]],
       [[   3.44008235,    1.69538493],
        [   1.69538493,    2.53472152]],
       [[   3.42258652,    1.68517428],
        [   1.68517428,    2.52876314]],
       [[   3.42023228,    1.68400688],
        [   1.68400688,    2.5281844 ]],
       [[   3.42002527,    1.6839348 ],
        [   1.6839348 ,    2.52815936]],
       [[   3.42001882,    1.68393592],
        [   1.68393592,    2.52815922]]])
        
        true_flt_mn =np.array([[ 14.93815212,   0.        ],
       [ 20.64331037,   7.78504301],
       [ 25.45836658,   7.37845042],
       [ 26.49336392,   5.38150366],
       [ 28.93090196,   5.20541277],
       [ 26.88122405,   3.02256071],
       [ 28.72372285,   3.74558702],
       [ 29.92435021,   3.88263335],
       [ 33.05306473,   4.9597845 ],
       [ 33.89896863,   4.52967857]])

        true_flt_vr = np.array([[[   0.99009901,    0.        ],
        [   0.        ,  100.        ]],
       [[   0.98538268,    1.06560283],
        [   1.06560283,    4.31755338]],
       [[   0.86081255,    0.55823003],
        [   0.55823003,    2.25836151]],
       [[   0.79971433,    0.42030185],
        [   0.42030185,    1.94726441]],
       [[   0.77964681,    0.38782167],
        [   0.38782167,    1.89471793]],
       [[   0.77477895,    0.38183637],
        [   0.38183637,    1.8873619 ]],
       [[   0.77388797,    0.38103817],
        [   0.38103817,    1.88664741]],
       [[   0.77376754,    0.38097701],
        [   0.38097701,    1.88661649]],
       [[   0.77375695,    0.38097855],
        [   0.38097855,    1.88661633]],
       [[   0.77375662,    0.38097936],
        [   0.38097936,    1.8866144 ]]])
        
        np.random.seed(0)
        state, observ = self.model.simulate_data(self.K)
        flt, prd, lhood = self.model.kalman_filter(observ)
        
        np.testing.assert_almost_equal(flt.mn,true_flt_mn)
        np.testing.assert_almost_equal(flt.vr,true_flt_vr)
        np.testing.assert_almost_equal(prd.mn,true_prd_mn)
        np.testing.assert_almost_equal(prd.vr,true_prd_vr)


class SmootherTestCase(LinearModelOperationsTestCase):
    def runTest(self):

        true_smt_mn = np.array([[ 15.25478089,   8.16168127],
       [ 20.69559248,   7.11640343],
       [ 24.72456361,   5.8493775 ],
       [ 26.44311329,   5.13963541],
       [ 27.96497005,   4.48424331],
       [ 27.50228336,   5.04726022],
       [ 29.17714049,   5.36323359],
       [ 30.80565571,   5.5569894 ],
       [ 33.09237748,   5.03297619],
       [ 33.89896863,   4.52967857]])
        
        true_smt_vr = np.array([[[ 0.84212747, -0.47046419],
        [-0.47046419,  1.36093023]],
       [[ 0.5485336 , -0.07533884],
        [-0.07533884,  0.83231186]],
       [[ 0.54774982, -0.06397322],
        [-0.06397322,  0.70775746]],
       [[ 0.54235149, -0.07488472],
        [-0.07488472,  0.68605901]],
       [[ 0.5383068 , -0.0784352 ],
        [-0.0784352 ,  0.68349283]],
       [[ 0.53735838, -0.07998772],
        [-0.07998772,  0.68531949]],
       [[ 0.53810584, -0.084037  ],
        [-0.084037  ,  0.70020252]],
       [[ 0.538553  , -0.09038111],
        [-0.09038111,  0.7801361 ]],
       [[ 0.54525736, -0.04443734],
        [-0.04443734,  1.09458568]],
       [[ 0.77375662,  0.38097936],
        [ 0.38097936,  1.8866144 ]]])

        np.random.seed(0)
        state, observ = self.model.simulate_data(self.K)
        flt, prd, lhood = self.model.kalman_filter(observ)
        smt = self.model.rts_smoother(flt, prd)
        
        np.testing.assert_almost_equal(smt.mn,true_smt_mn)
        np.testing.assert_almost_equal(smt.vr,true_smt_vr)


class SamplerTestCase(LinearModelOperationsTestCase):
    def runTest(self):
        
        true_smp = np.array([[ 16.76187561,   6.85037407],
       [ 21.91276618,   6.65131843],
       [ 25.44617624,   5.59531396],
       [ 28.24896593,   4.31521661],
       [ 27.7783341 ,   3.35303131],
       [ 27.50559835,   5.15217254],
       [ 29.30521224,   6.26581596],
       [ 31.00831517,   4.75796016],
       [ 32.67351594,   3.90126029],
       [ 34.25639136,   4.6486885 ]])
       
        np.random.seed(0)
        state, observ = self.model.simulate_data(self.K)
        smp = self.model.sample_posterior(observ)
        
        np.testing.assert_almost_equal(smp,true_smp)

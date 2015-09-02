import unittest

import numpy as np

from kalman import GaussianDensity
from linear_models import DegenerateLinearModel
from learners_mcmc import (
    BaseMCMCLearner, 
    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
    MCMCLearnerTransitionDegenerateModelWithMNIWPrior)

# Create a learner class for this test
class MCMCLearner(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionDegenerateModelWithMNIWPrior):
    pass

class DegenerateModelLearningTestCase(unittest.TestCase):
    def setUp(self):

        # Basic parameters
        self.K = 100
        self.ds = 3
        self.do = 3

        # System matrices
        params = dict()
        params['F'] = np.array([[0.9,0.8,0.7],[0,0.9,0.8],[0,0,0.7]])
        params['rank'] = np.array([2])
        params['vec'] = (1./np.sqrt(3))*np.array([[1,1],[1,1],[1,-1]])
        params['val'] = np.array([1./5,1./2])
        params['H'] = np.identity(self.do)
        params['R'] = 0.1*np.identity(self.do)
        self.params = params

        # Create model
        prior = GaussianDensity(np.zeros(self.ds), np.identity(self.ds))
        self.model = DegenerateLinearModel(self.ds, self.do, prior, 
                                                                  self.params)

        # Simulate data
        np.random.seed(1)
        self.state, self.observ = self.model.simulate_data(self.K)

        # Create initial estimated model
        est_params = dict()
        est_params['F'] = 0.5*np.identity(self.ds)
        est_params['rank'] = np.array([2])
        est_params['vec'] = np.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])
        est_params['val'] = np.array([1,1])
        est_params['H'] = np.identity(self.do)
        est_params['R'] = np.identity(self.do)
        est_model = DegenerateLinearModel(self.ds, self.do, prior, est_params)
        self.est_model = est_model

        # Set MCMC parameters
        self.num_iter = 2000
        self.num_burn = int(self.num_iter/5)


class MNIWPriorLearning(DegenerateModelLearningTestCase):
    def runTest(self):
        
        # Algorithm parameters
        num_hold = int(self.num_burn/4)
        algoparams = dict()
        algoparams['rotate'] = 0.1
        algoparams['perturb'] = 0.1
        
        # Set up learning
        hyperparams = dict()
        hyperparams['nu0'] = self.ds-1
        hyperparams['Psi0'] = hyperparams['nu0']*np.identity(self.ds)
        hyperparams['M0'] = np.zeros(self.ds)
        hyperparams['V0'] = np.identity(self.ds)
        hyperparams['a0'] = 1
        hyperparams['b0'] = hyperparams['a0']*0.1
        learner = MCMCLearner(self.est_model, self.observ, hyperparams,
                                                        algoparams=algoparams)
        
        # MCMC
        for ii in range(self.num_iter):
            print("Running iteration {} of {}.".format(ii+1,self.num_iter))
        
            if (ii%3)==0:
                learner.sample_transition_covariance('rotate')
            elif (ii%3)==1:
                learner.sample_transition_covariance('rank')
            else:
                learner.sample_transition_matrix()
            learner.sample_transition_within_subspace()
            if ii > num_hold:
                learner.sample_observation_diagonal_covariance()
            learner.sample_state_trajectory()
            learner.save_link()
            
            if ((ii+1)%20)==0:
                learner.adapt_algorithm_parameters()
            
        learner.plot_chain_trace('F',
                                 numBurnIn=self.num_burn,
                                 trueModel=self.model)

        learner.plot_chain_trace('transition_covariance',
                                 numBurnIn=self.num_burn,
                                 trueModel=self.model,
                                 derived=True)

        learner.plot_chain_trace('rank',
                                 numBurnIn=self.num_burn,
                                 trueModel=self.model)

        learner.plot_chain_trace('R',
                                 numBurnIn=self.num_burn,
                                 trueModel=self.model)
        
        learner.plot_chain_histogram('F',
                                     numBurnIn=self.num_burn,
                                     trueModel=self.model)
                                     
        learner.plot_chain_histogram('transition_covariance',
                                     numBurnIn=self.num_burn, 
                                     trueModel=self.model,
                                     derived=True)
                                     
        learner.plot_chain_histogram('rank',
                                     numBurnIn=self.num_burn,
                                     trueModel=self.model)
                                     
        learner.plot_chain_histogram('R',
                                     numBurnIn=self.num_burn,
                                     trueModel=self.model)
        
        learner.plot_chain_acf('F', numBurnIn=self.num_burn)
        learner.plot_chain_acf('R', numBurnIn=self.num_burn)
        
        learner.plot_chain_accept()
        learner.plot_chain_adapt()
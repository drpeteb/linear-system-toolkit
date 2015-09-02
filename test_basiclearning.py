import unittest

import numpy as np

from kalman import GaussianDensity
from linear_models import BasicLinearModel
from learners_mcmc import (
    BaseMCMCLearner, 
    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
    MCMCLearnerTransitionBasicModelWithMNIWPrior)

# Create a learner class for this test
class MCMCLearner(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionBasicModelWithMNIWPrior):
    pass

class BasicModelLearningTestCase(unittest.TestCase):
    def setUp(self):

        # Basic parameters
        self.K = 100
        self.ds = 3
        self.do = 3

        # System matrices
        params = dict()
        params['F'] = np.array([[0.9,0.8,0.7],[0,0.9,0.8],[0,0,0.7]])
        params['Q'] = np.array([[0.1,0.05,0],[0.05,0.1,0.05],[0,0.05,0.1]])
        params['H'] = np.identity(self.do)
        params['R'] = 0.1*np.identity(self.do)
        self.params = params

        # Create model
        prior = GaussianDensity(np.zeros(self.ds), np.identity(self.ds))
        self.model = BasicLinearModel(self.ds, self.do, prior, self.params)

        # Simulate data
        np.random.seed(1)
        self.state, self.observ = self.model.simulate_data(self.K)

        # Create initial estimated model
        est_params = dict()
        est_params['F'] = 0.5*np.identity(self.ds)
        est_params['Q'] = np.identity(self.ds)
        est_params['H'] = np.identity(self.do)
        est_params['R'] = np.identity(self.do)
        est_model = BasicLinearModel(self.ds, self.do, prior, est_params)
        self.est_model = est_model

        # Set MCMC parameters
        self.num_iter = 200
        self.num_burn = int(self.num_iter/5)


class MNIWPriorLearning(BasicModelLearningTestCase):
    def runTest(self):

        # Set up learning
        hyperparams = dict()
        hyperparams['nu0'] = self.ds-1
        hyperparams['Psi0'] = hyperparams['nu0']*0.1*np.identity(self.ds)
        hyperparams['M0'] = np.zeros(self.ds)
        hyperparams['V0'] = np.identity(self.ds)
        hyperparams['a0'] = 1
        hyperparams['b0'] = hyperparams['a0']*0.1
        learner = MCMCLearner(self.est_model, self.observ, hyperparams)

        # MCMC
        for ii in range(self.num_iter):
            print("Running iteration {} of {}.".format(ii+1,self.num_iter))
        
            learner.sample_transition()
            learner.sample_observation_diagonal_covariance()
            learner.sample_state_trajectory()
            learner.save_link()
        
        learner.plot_chain_trace('F', numBurnIn=self.num_burn,
                                                         trueModel=self.model)
        learner.plot_chain_trace('Q', numBurnIn=self.num_burn,
                                                         trueModel=self.model)
        learner.plot_chain_trace('R', numBurnIn=self.num_burn,
                                                         trueModel=self.model)
        
        
        learner.plot_chain_histogram('F', numBurnIn=self.num_burn,
                                                         trueModel=self.model)
        learner.plot_chain_histogram('Q', numBurnIn=self.num_burn, 
                                                         trueModel=self.model)
        learner.plot_chain_histogram('R', numBurnIn=self.num_burn,
                                                         trueModel=self.model)

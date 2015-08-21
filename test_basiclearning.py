import unittest

import numpy as np

from kalman import GaussianDensity
from linear_models import BasicLinearModel
from learners_mcmc import MCMCLearnerForBasicModelWithMNIWPrior, MCMCLearnerForBasicModelWithIndependentPriors

class BasicModelLearningTestCase(unittest.TestCase):
    def setUp(self):
        
        # Basic parameters
        self.K = 10
        self.ds = 2
        self.do = 2
        
        # System matrices
        params = dict()
        params['F'] = np.array([[0.9,0.81],[0,0.9]])
        params['Q'] = np.array([[0.1,0],[0,0.1]])
        params['H'] = np.array([[1,0],[0,1]])
        params['R'] = np.array([[0.1,0],[0,0.1]])
        self.params = params
        
        # Create model
        prior = GaussianDensity(np.array([0,0]), np.array([[100,0],[0,100]]))
        self.model = BasicLinearModel(self.ds, self.do, prior, params)
        
        # Simulate data
        np.random.seed(0)
        self.state, self.observ = self.model.simulate_data(self.K)
        
        # Create initial estimated model
        est_params = dict()
        est_params['F'] = np.array([[0.5,0],[0,0.5]])
        est_params['Q'] = 0.3*np.identity(self.ds)
        est_params['H'] = np.identity(self.do)#np.array([[1,0]])
        est_params['R'] = np.identity(self.do)#np.array([[1]])
        est_model = BasicLinearModel(self.ds, self.do, prior, est_params)
        self.est_model = est_model
        
        # Set MCMC parameters
        self.num_iter = 200
        self.num_burn = int(self.num_iter/5)


class MNIWPriorLearning(BasicModelLearningTestCase):
    def runTest(self):
        
        # Set up learning
        hyperparams = dict()
        hyperparams['nu0'] = 3.01
        hyperparams['Psi0'] = (hyperparams['nu0']-self.ds-1)*np.identity(self.ds)
        hyperparams['M0'] = np.zeros(self.ds)
        hyperparams['V0'] = np.identity(self.ds)
        learner = MCMCLearnerForBasicModelWithMNIWPrior(self.est_model, self.observ, hyperparams)
        
        # MCMC
        for ii in range(self.num_iter):
            print("Running iteration {} of {}.".format(ii+1,self.num_iter))
            
            learner.iterate_transition()
            learner.save_link()
        
        learner.plot_chain_trace('F', numBurnIn=self.num_burn)
        learner.plot_chain_trace('Q', numBurnIn=self.num_burn)
        
        learner.plot_chain_histogram('F', numBurnIn=self.num_burn, trueValue=self.params['F'])
        learner.plot_chain_histogram('Q', numBurnIn=self.num_burn, trueValue=self.params['Q'])
        
        
class IndependentPriorsLearning(BasicModelLearningTestCase):
    def runTest(self):
        
        # Set up learning
        hyperparams = dict()
        hyperparams['nu0'] = 3.01
        hyperparams['Psi0'] = (hyperparams['nu0']-self.ds-1)*np.identity(self.ds)
        hyperparams['M0'] = np.zeros(self.ds)
        hyperparams['alpha'] = 100
        learner = MCMCLearnerForBasicModelWithIndependentPriors(self.est_model, self.observ, hyperparams)
        
        # MCMC
        for ii in range(self.num_iter):
            print("Running iteration {} of {}.".format(ii+1,self.num_iter))
            
            learner.iterate_transition()
            learner.save_link()
        
        learner.plot_chain_trace('F', numBurnIn=self.num_burn)
        learner.plot_chain_trace('Q', numBurnIn=self.num_burn)
        
        learner.plot_chain_histogram('F', numBurnIn=self.num_burn, trueValue=self.params['F'])
        learner.plot_chain_histogram('Q', numBurnIn=self.num_burn, trueValue=self.params['Q'])
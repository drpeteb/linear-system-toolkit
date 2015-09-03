from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from kalman import GaussianDensity
from linear_models import BasicLinearModel
from learners_mcmc import (
    BaseMCMCLearner,
    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
    MCMCLearnerTransitionBasicModelWithMNIWPrior)

# Create a learner class for this run
class MCMCLearner(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionBasicModelWithMNIWPrior):
    pass


plt.close('all')
filename = 'mcmc-toy-basic'

K = 200
ds = 3
do = 3

params = dict()
params['F'] = np.array([[0.9,0.8,0.7],[0,0.9,0.8],[0,0,0.7]])#0.9*np.identity(ds)#
params['Q'] = np.array([[0.1,0.05,0],[0.05,0.1,0.05],[0,0.05,0.1]])
params['H'] = np.identity(do)#np.array([[1,0,0],[0,1,0]])#
params['R'] = 0.1*np.identity(do)#np.array([[1]])

prior = GaussianDensity(np.zeros(ds), 100*np.identity(ds))
model = BasicLinearModel(ds, do, prior, params)

np.random.seed(1)
state, observ = model.simulate_data(K)

est_params = deepcopy(params)
est_params['F'] = 0.5*np.identity(ds)
est_params['Q'] = np.identity(ds)
est_params['R'] = np.identity(do)
est_model = BasicLinearModel(ds, do, prior, est_params)

hyperparams = dict()
hyperparams['nu0'] = ds-1
hyperparams['Psi0'] = hyperparams['nu0']*0.1*np.identity(ds)
hyperparams['M0'] = np.zeros((ds,ds))
hyperparams['V0'] = np.identity(ds)
hyperparams['a0'] = 1
hyperparams['b0'] = hyperparams['a0']*0.1

algoparams = dict()
algoparams['Qs'] = 0.1
algoparams['Fs'] = 0.1

learner = MCMCLearner(est_model, observ, hyperparams, algoparams=algoparams, verbose=True)

num_iter = 200
num_burn = int(num_iter/5)

for ii in range(num_iter):
    print("Running iteration {} of {}.".format(ii+1,num_iter))
    
    learner.sample_transition()
    learner.sample_observation_diagonal_covariance()
    learner.sample_state_trajectory()
    learner.save_link()

learner.plot_chain_trace('F', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_trace('Q', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_trace('R', numBurnIn=num_burn, trueModel=model)


learner.plot_chain_histogram('F', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_histogram('Q', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_histogram('R', numBurnIn=num_burn, trueModel=model)

learner.save(filename)
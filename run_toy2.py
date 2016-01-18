from copy import deepcopy
import pickle

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt

from kalman import GaussianDensity
from linear_models import BasicLinearModel, DegenerateLinearModel
from learners_mcmc import (
    BaseMCMCLearner,
    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
    MCMCLearnerTransitionBasicModelWithMNIWPrior,
    MCMCLearnerTransitionDegenerateModelWithMNIWPrior)

# Create a learner class for this run
class MCMCLearnerB(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionBasicModelWithMNIWPrior):
    pass

class MCMCLearnerD(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionDegenerateModelWithMNIWPrior):
    pass


plt.close('all')
np.random.seed(0)

test_data_file = 'toy2-test-data.p'
model_type = 'degenerate'
K = 100
num_warm = 100
num_iter = 10000
num_burn = int(num_iter/2)


ds = 6
do = 6
params = dict()
params['F'] = 0.99*np.identity(ds)
params['rank'] = np.array([3])

A = np.round(np.random.randn(ds,params['rank']),2)
print(A)
Q = np.dot(A,A.T)
params['val'], params['vec'] = la.eigh(Q)
params['Q'] = Q

params['H'] = np.identity(do)
params['R'] = 10*np.identity(do)
prior = GaussianDensity(np.zeros(ds), 10*np.identity(ds))

model = DegenerateLinearModel(ds, do, prior, params)
state, observ = model.simulate_data(K)

# Draw it
fig, axs = plt.subplots(nrows=ds,ncols=1)
for dd in range(ds):
    axs[dd].plot(observ[:,dd])

# Save the data
fh = open(test_data_file, 'wb')
pickle.dump([model, state, observ], fh)
fh.close()

est_params = deepcopy(params)
est_params['F'] = 0.5*np.identity(ds)
est_params['rank'] = np.array([ds])
est_params['val'] = 1 + 0.2*np.random.randn(ds)
est_params['vec'] = np.identity(ds)
est_params['Q'] = np.dot(est_params['vec'], np.dot(np.diag(est_params['val']), est_params['vec'].T))
est_params['R'] = np.identity(do)

hyperparams = dict()
hyperparams['nu0'] = ds-1
hyperparams['rPsi0'] = np.identity(ds)
hyperparams['Psi0'] = hyperparams['nu0']*hyperparams['rPsi0']
hyperparams['alpha'] = 1
hyperparams['M0'] = np.zeros((ds,ds))
hyperparams['V0'] = 100*np.identity(ds)
hyperparams['a0'] = 1
hyperparams['b0'] = hyperparams['a0']*0.1

algoparams = dict()
algoparams['rotate'] = 1E-2
algoparams['perturb'] = 1E-6

# Learning
if model_type == 'basic':

    est_model = BasicLinearModel(ds, do, prior, est_params)
    learner = MCMCLearnerB(est_model, observ, hyperparams, algoparams=algoparams, verbose=True)

    for ii in range(num_iter):
        print("Running iteration {} of {}.".format(ii+1,num_iter))

        learner.sample_transition()
        learner.sample_observation_diagonal_covariance()
        learner.sample_state_trajectory()
        learner.save_link()

elif model_type == 'degenerate':

    est_model = DegenerateLinearModel(ds, do, prior, est_params)
    learner = MCMCLearnerD(est_model, observ, hyperparams, algoparams=algoparams, verbose=True)

    for ii in range(num_warm):
        print("Running warm-up iteration {} of {}.".format(ii+1,num_warm))
        learner.sample_transition_within_subspace()
        learner.sample_observation_diagonal_covariance()
        learner.sample_state_trajectory()

    for ii in range(num_iter):
        print("")
        print("Running iteration {} of {}.".format(ii+1,num_iter))

        move_prob = np.array([0.2,0.4,0.4])
        u = np.random.choice(3, p=move_prob)

        if u == 0:
            learner.sample_transition_covariance('rank')
        elif u == 1:
            learner.sample_transition_covariance('rotate')
        elif u == 2:
            learner.sample_transition_matrix()
        learner.sample_transition_within_subspace()
        learner.sample_observation_diagonal_covariance()
        learner.sample_state_trajectory()
        learner.save_link()
        print("Current rank: {}".format(learner.model.parameters['rank'][0]))

        if ((ii+1)%20)==0:
            learner.adapt_algorithm_parameters()

learner.plot_chain_trace('F', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_trace('Q', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_trace('R', numBurnIn=num_burn, trueModel=model)


learner.plot_chain_histogram('F', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_histogram('Q', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_histogram('R', numBurnIn=num_burn, trueModel=model)

learner.plot_chain_acf('F', numBurnIn=num_burn)
learner.plot_chain_acf('R', numBurnIn=num_burn)

if model_type == 'degenerate':
    learner.plot_chain_trace('rank', numBurnIn=num_burn, trueModel=model)
    learner.plot_chain_accept()
    learner.plot_chain_adapt()

filename = './results/toy2-mcmc-{}.p'.format(model_type)
learner.save(filename)
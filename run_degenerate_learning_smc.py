from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from kalman import GaussianDensity
from linear_models import DegenerateLinearModel
from learners_mcmc import (
    BaseMCMCLearner,
    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
    MCMCLearnerTransitionDegenerateModelWithMNIWPrior)
from learners_smc import (DegenerateSMCLearner,
                          effective_sample_size,
                          normalising_constant_estimate)

# Create a learner class for this run
class MCMCLearner(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionDegenerateModelWithMNIWPrior):
    pass


plt.close('all')
filename = './results/toy-smc-degenerate.p'

K = 100
ds = 3
do = 3

params = dict()
params['F'] = np.array([[0.9,0.8,0.7],[0,0.9,0.8],[0,0,0.7]])
params['rank'] = np.array([2])
params['vec'] = (1./np.sqrt(3))*np.array([[1,1],[1,1],[1,-1]])
params['val'] = np.array([1./5,1./2])
params['H'] = np.identity(do)
params['R'] = 0.1*np.identity(do)

prior = GaussianDensity(np.zeros(ds), 100*np.identity(ds))
model = DegenerateLinearModel(ds, do, prior, params)

np.random.seed(0)
state, observ = model.simulate_data(K)

est_params = deepcopy(params)
est_params['F'] = 0.5*np.identity(ds)
est_params['rank'] = np.array([ds])
est_params['vec'] = np.identity(ds)
est_params['val'] = np.array([1./3, 1./4, 1./5])
est_params['R'] = np.identity(do)
est_model = DegenerateLinearModel(ds, do, prior, est_params)

hyperparams = dict()
#hyperparams['nu0'] = params['rank']
hyperparams['rPsi0'] = np.identity(ds)
hyperparams['M0'] = np.zeros((ds,ds))
hyperparams['V0'] = np.identity(ds)
hyperparams['a0'] = 1
hyperparams['b0'] = 0.2
hyperparams['alpha'] = 0.001

algoparams = dict()

learner = MCMCLearner(est_model, observ, hyperparams, algoparams=algoparams, verbose=True)

num_iter = 1000
num_burn = int(num_iter/2)

# MCMC with full rank
for ii in range(num_iter):
    print("Running iteration {} of {}.".format(ii+1,num_iter))
    learner.sample_transition_within_subspace()
    learner.sample_observation_diagonal_covariance()
    learner.sample_state_trajectory()
    learner.save_link()

num_rejuv = 5

# Use Markov Chain to initialise SMC
smclearner = DegenerateSMCLearner(learner.chain_model[num_burn:],
                                  learner.chain_lhood[num_burn:],
                                  learner.chain_state[num_burn:],
                                  observ,
                                  hyperparams,
                                  prior,
                                  num_rejuv,
                                  verbose=True)

mlhoodratios = np.zeros(ds)
effsampsizes = np.zeros(ds)
for rr in reversed(range(1,ds)):
    print("")
    print("Running SMC sampler for rank {}.".format(rr))
    smclearner.smc_reduce_rank(rr)
    mlhoodratios[rr] = normalising_constant_estimate(smclearner.approx[rr].weight)
    effsampsizes[rr] = effective_sample_size(smclearner.approx[rr].weight)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(ds), mlhoodratios)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(ds), effsampsizes)
plt.show()

#learner.plot_chain_trace('F', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_trace('transition_covariance', numBurnIn=num_burn, trueModel=model, derived=True)
#learner.plot_chain_trace('rank', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_trace('R', numBurnIn=num_burn, trueModel=model)
#
#learner.plot_chain_histogram('F', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_histogram('transition_covariance', numBurnIn=num_burn, trueModel=model, derived=True)
#learner.plot_chain_histogram('rank', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_histogram('R', numBurnIn=num_burn, trueModel=model)
#
#learner.plot_chain_acf('F', numBurnIn=num_burn)
#learner.plot_chain_acf('R', numBurnIn=num_burn)
#
#learner.plot_chain_accept()
#learner.plot_chain_adapt()

learner.save(filename)
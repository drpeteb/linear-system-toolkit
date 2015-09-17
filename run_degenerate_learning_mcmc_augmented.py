from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from kalman import GaussianDensity
from linear_models import DegenerateLinearModel
from learners_mcmc_augmented import CleverDegenerateMCMCLearner

plt.close('all')
filename = './results/toy-mcmc-degenerate.p'

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
hyperparams['nu0'] = params['rank']
hyperparams['rPsi0'] = np.identity(ds)
hyperparams['M0'] = np.zeros((ds,ds))
hyperparams['V0'] = np.identity(ds)
hyperparams['a0'] = 1
hyperparams['b0'] = 0.2
hyperparams['alpha'] = 0.001

algoparams = dict()
algoparams['rotate'] = 1E-1
algoparams['perturb'] = 1E-6

learner = CleverDegenerateMCMCLearner(est_model, observ, hyperparams,
                                      algoparams, verbose=True)

num_iter = 200
num_burn = int(num_iter/2)
num_warm = 20

for ii in range(num_warm):
    learner.sample_transition_warm_up()

for ii in range(num_iter):
    print("")
    print("Running iteration {} of {}.".format(ii+1,num_iter))

    rchange = np.random.random_integers(-1,1)
    rank = learner.model.parameters['rank'][0] + rchange
    rank = max(rank,1)
    rank = min(rank,ds)
    learner.sample_transition(rank)

    learner.save_link()
    print("Current rank: {}".format(learner.model.parameters['rank'][0]))

#    if ((ii+1)%20)==0:
#        learner.adapt_algorithm_parameters()

learner.plot_chain_trace('F', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_trace('transition_covariance', numBurnIn=num_burn, trueModel=model, derived=True)
learner.plot_chain_trace('rank', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_trace('R', numBurnIn=num_burn, trueModel=model)

learner.plot_chain_histogram('F', numBurnIn=num_burn, trueModel=model)
#learner.plot_chain_histogram('transition_covariance', numBurnIn=num_burn, trueModel=model, derived=True)
learner.plot_chain_histogram('rank', numBurnIn=num_burn, trueModel=model)
learner.plot_chain_histogram('R', numBurnIn=num_burn, trueModel=model)

learner.plot_chain_acf('F', numBurnIn=num_burn)
learner.plot_chain_acf('R', numBurnIn=num_burn)

learner.plot_chain_accept()
learner.plot_chain_adapt()

learner.save(filename)
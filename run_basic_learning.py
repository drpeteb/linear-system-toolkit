from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt

from kalman import GaussianDensity
from linear_models import BasicLinearModel, DegenerateLinearModel
from learners_mcmc import MCMCLearnerForDegenerateModelWithMNIWPrior

plt.close('all')

K = 100
ds = 3
do = 3

params = dict()
params['F'] = np.array([[0.9,0.8,0.7],[0,0.9,0.8],[0,0,0.7]])
params['rank'] = 2
params['vec'] = (1./np.sqrt(3))*np.array([[1,1],[1,1],[1,-1]])#np.identity(ds)
params['val'] = np.array([5,2])
#params['Q'] = np.array([[0.1,0.1],[0.1,0.1]])#np.array([[0.1,0.05],[0.05,0.1]])
params['H'] = np.identity(do)#np.array([[1,0]])
params['R'] = 0.1*np.identity(do)#np.array([[1]])

prior = GaussianDensity(np.zeros(ds), 100*np.identity(ds))
model = DegenerateLinearModel(ds, do, prior, params)

np.random.seed(0)
state, observ = model.simulate_data(K)

est_params = dict(params)
est_model = DegenerateLinearModel(ds, do, prior, est_params)

hyperparams = dict()
hyperparams['nu0'] = params['rank']
hyperparams['Psi0'] = params['rank']*np.identity(ds)
hyperparams['M0'] = np.zeros((ds,ds))
hyperparams['V0'] = np.identity(ds)
learner = MCMCLearnerForDegenerateModelWithMNIWPrior(est_model, observ, hyperparams)

num_iter = 200
num_burn = int(num_iter/5)

for ii in range(num_iter):
    print("Running iteration {} of {}.".format(ii+1,num_iter))
    
    learner.iterate_transition()
    learner.save_link()

learner.plot_chain_trace('F', numBurnIn=num_burn)
#learner.plot_chain_trace('Q', numBurnIn=num_burn)
learner.plot_chain_trace('val', numBurnIn=num_burn)
learner.plot_chain_trace('vec', numBurnIn=num_burn)

learner.plot_chain_histogram('F', numBurnIn=num_burn, trueValue=params['F'])
#learner.plot_chain_histogram('Q', numBurnIn=num_burn, trueValue=params['Q'])
learner.plot_chain_histogram('val', numBurnIn=num_burn, trueValue=params['val'])
learner.plot_chain_histogram('vec', numBurnIn=num_burn, trueValue=params['vec'])


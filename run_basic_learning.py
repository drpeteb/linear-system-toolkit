from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt

from basic import GaussianDensity
from linear_models import BasicLinearModel
from learners_mcmc import MCMCLearnerForBasicModelWithMNIWPrior, MCMCLearnerForBasicModelWithIndependentPriors

plt.close('all')

K = 100
ds = 2
do = 2

params = dict()
params['F'] = np.array([[0.9,0.81],[0,0.9]])
params['Q'] = np.array([[0.1,0.05],[0.05,0.1]])
params['H'] = np.identity(do)#np.array([[1,0]])
params['R'] = 0.1*np.identity(do)#np.array([[1]])

prior = GaussianDensity(np.array([0,0]), np.array([[100,0],[0,100]]))
model = BasicLinearModel(ds, do, prior, params)

np.random.seed(1)
state, observ = model.simulate_data(K)

est_params = dict()
est_params['F'] = np.array([[0.5,0],[0,0.5]])
est_params['Q'] = 0.3*np.identity(ds)
est_params['H'] = np.identity(do)#np.array([[1,0]])
est_params['R'] = np.identity(do)#np.array([[1]])
est_model = BasicLinearModel(ds, do, prior, est_params)

#hyperparams = dict()
#hyperparams['nu0'] = 4
#hyperparams['Psi0'] = 0.1*np.identity(ds)
#hyperparams['M0'] = np.zeros(ds)
#hyperparams['V0'] = np.identity(ds)
#learner = MCMCLearnerForBasicModelWithMNIWPrior(est_model, observ, hyperparams)

hyperparams = dict()
hyperparams['nu0'] = 4
hyperparams['Psi0'] = (hyperparams['nu0']-ds-1)*np.identity(ds)
hyperparams['M0'] = np.zeros(ds)
hyperparams['alpha'] = 100
learner = MCMCLearnerForBasicModelWithIndependentPriors(est_model, observ, hyperparams)

num_iter = 500

for ii in range(num_iter):
    print("Running iteration {} of {}.".format(ii+1,num_iter))
    
    learner.iterate_transition()
    learner.save_link()



fig = plt.figure()
cnt = 0
for ii in range(ds):
    for jj in range(ds):
        cnt += 1
        ax = fig.add_subplot(ds,ds,cnt)
        ax.plot([mm.transition_matrix()[ii,jj] for mm in learner.chain])

fig = plt.figure()
cnt = 0
for ii in range(ds):
    for jj in range(ds):
        cnt += 1
        ax = fig.add_subplot(ds,ds,cnt)
        ax.plot([mm.transition_covariance()[ii,jj] for mm in learner.chain])
        
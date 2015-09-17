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
from learners_smc import (DegenerateSMCLearner,
                          effective_sample_size,
                          normalising_constant_estimate)

# Create a learner class for this run
class MCMCDegenerateLearner(
                BaseMCMCLearner,
                MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                MCMCLearnerTransitionDegenerateModelWithMNIWPrior):
    pass

plt.close('all')
np.random.seed(0)

data_path = './mocap-data/'
test_data_file = './results/mocap-test-data.p'

num_iter = 750
num_burn = 500
thin = 5
num_rejuv = 2

# Import marker data
markers = np.genfromtxt(data_path+'downsampled_head_markers.csv', delimiter=',')
truth = np.genfromtxt(data_path+'downsampled_head_markers_truth.csv', delimiter=',')
K,d = markers.shape
num_markers = int(d/3)

# Knock out a section to test on
test_sections = [np.arange(40,60), np.arange(90,110),
                                       np.arange(140,160), np.arange(190,210)]

# Missing data
missing_marker = 0
for sec in test_sections:
    for kk in sec:
        markers[kk][3*missing_marker:3*(missing_marker+1)] = np.NaN
    missing_marker += 1

# Save the data
fh = open(test_data_file, 'wb')
pickle.dump(markers, fh)
fh.close()

# Draw it
fig, axs = plt.subplots(nrows=3,ncols=1)
color_list = 'brgc'
for mm in range(num_markers):
    for dd in range(3):
        axs[dd].plot(markers[:,3*mm+dd], color=color_list[mm])

# Set up model
ds = 2*d
do = d

# Initial estimate
est_params = dict()
Imat = np.identity(d)
Zmat = np.zeros((d,d))


est_params['F'] = np.vstack((np.hstack((Imat,Imat)), np.hstack((Zmat,Imat))))
est_params['Q'] = np.vstack((np.hstack((Imat,Imat)), np.hstack((Imat,Imat))))
est_params['val'], est_params['vec'] = la.eigh(est_params['Q'])
est_params['rank'] = np.array([ds])
est_params['H'] = np.hstack((np.identity(d),np.zeros((d,d))))
est_params['R'] = 0.001*np.identity(d)

prior = GaussianDensity(np.zeros(ds), 1000*np.identity(ds))
est_degenerate_model = DegenerateLinearModel(ds, do, prior, est_params)
est_basic_model = BasicLinearModel(ds, do, prior, est_params)
est_naive_model = BasicLinearModel(ds, do, prior, est_params)

# Hyperparameters
hyperparams = dict()
hyperparams['rPsi0'] = np.identity(ds)
hyperparams['M0'] = np.zeros((ds,ds))
hyperparams['V0'] = 100*np.identity(ds)
hyperparams['a0'] = 1
hyperparams['b0'] = 0.001

# Algorithm parameters
algoparams = dict()

# Create the MCMC object
learner = MCMCDegenerateLearner(est_degenerate_model, markers,
                        hyperparams, algoparams=algoparams, verbose=True)

for ii in range(num_iter):
    print("Running iteration {} of {}.".format(ii+1,num_iter))

    learner.sample_transition_within_subspace()
    learner.sample_observation_diagonal_covariance()
    learner.sample_state_trajectory()
    learner.save_link()


# Use Markov Chain to initialise SMC
smclearner = DegenerateSMCLearner(learner.chain_model[num_burn::thin],
                                  learner.chain_lhood[num_burn::thin],
                                  learner.chain_state[num_burn::thin],
                                  markers,
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

filename = "./results/mocap-smc-degenerate.p"
learner.save(filename)
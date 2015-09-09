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
class MCMCBasicLearner(
                BaseMCMCLearner,
                MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                MCMCLearnerTransitionBasicModelWithMNIWPrior):
    pass

class MCMCDegenerateLearner(
                BaseMCMCLearner,
                MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                MCMCLearnerTransitionDegenerateModelWithMNIWPrior):
    pass

class MCMCNaiveLearner(
                BaseMCMCLearner,
                MCMCLearnerObservationDiagonalCovarianceWithIGPrior):
    pass

plt.close('all')
np.random.seed(0)

data_path = './mocap-data/'
test_data_file = './results/mocap-test-data.p'

model_type = 'degenerate'#'naive'#'basic'#
num_iter = 20000

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
#est_params['vec'] = np.identity(ds)
#est_params['val'] = np.ones(ds)
#est_params['Q'] = np.dot( est_params['vec'],
#                      np.dot(np.diag(est_params['val']),est_params['vec'].T) )
est_params['H'] = np.hstack((np.identity(d),np.zeros((d,d))))
est_params['R'] = 0.001*np.identity(d)

prior = GaussianDensity(np.zeros(ds), 1000*np.identity(ds))
est_degenerate_model = DegenerateLinearModel(ds, do, prior, est_params)
est_basic_model = BasicLinearModel(ds, do, prior, est_params)
est_naive_model = BasicLinearModel(ds, do, prior, est_params)

# Hyperparameters
hyperparams = dict()
hyperparams['nu0'] = ds
hyperparams['rPsi0'] = np.identity(ds)
hyperparams['M0'] = np.zeros((ds,ds))
hyperparams['V0'] = 100*np.identity(ds)
hyperparams['a0'] = 1
hyperparams['b0'] = 0.001

# Algorithm parameters
algoparams = dict()
algoparams['rotate'] = 0.001
algoparams['perturb'] = 0.000000001
                                          
num_burn = int(num_iter/2)
num_hold = min(100,int(num_burn/4))

if model_type == 'naive':
    
    # Create the MCMC object
    learner = MCMCNaiveLearner(est_naive_model, markers, 
                            hyperparams, algoparams=algoparams, verbose=True)
    
    # Naive model learning
    for ii in range(num_iter):
        print("Running iteration {} of {}.".format(ii+1,num_iter))
        
        if ii > num_hold:
            learner.sample_observation_diagonal_covariance()
        learner.sample_state_trajectory()
        learner.save_link()

elif model_type == 'basic':
    
    # Create the MCMC object
    learner = MCMCBasicLearner(est_basic_model, markers, 
                            hyperparams, algoparams=algoparams, verbose=True)
    
    # Basic model learning
    for ii in range(num_iter):
        print("Running iteration {} of {}.".format(ii+1,num_iter))
    
        learner.sample_transition()
        if ii > num_hold:
            learner.sample_observation_diagonal_covariance()
        learner.sample_state_trajectory()
        learner.save_link()
    
elif model_type == 'degenerate':
    
    # Create the MCMC object
    learner = MCMCDegenerateLearner(est_degenerate_model, markers, 
                            hyperparams, algoparams=algoparams, verbose=True)
    
    # Degenerate model learning
    for ii in range(10):
        learner.sample_transition_within_subspace()
    
    for ii in range(num_iter):
        print("")
        print("Running iteration {} of {}.".format(ii+1,num_iter))
        
        #learner.sample_transition_covariance('rank')
        learner.sample_transition_within_subspace()
        if ii > num_hold:
            learner.sample_observation_diagonal_covariance()
        if ii > 10:
            learner.sample_transition()
        learner.sample_state_trajectory()
        learner.save_link()
        print("Current rank: {}".format(learner.model.parameters['rank'][0]))
        
#        if (ii%3)==0:
#            learner.sample_transition_covariance('rotate')
#        elif (ii%3)==1:
#            learner.sample_transition_covariance('rank')
#        else:
#            learner.sample_transition_matrix()
#        learner.sample_transition_within_subspace()
#        if ii > num_hold:
#            learner.sample_observation_diagonal_covariance()
#        learner.sample_state_trajectory()
#        learner.save_link()
#        print("Current rank: {}".format(learner.model.parameters['rank'][0]))
#        
#        if ((ii+1)%20)==0:
#            learner.adapt_algorithm_parameters()
        
    learner.plot_chain_trace('rank', numBurnIn=num_burn)
    learner.plot_chain_accept()
    learner.plot_chain_adapt()

learner.plot_chain_trace('R', dims=([0],[0]))

state_mn, state_sd = learner.estimate_state_trajectory(numBurnIn=num_burn)
for mm in range(num_markers):
    for dd in range(3):
        axs[dd].plot(state_mn[:,3*mm+dd], '--', color=color_list[mm])
        axs[dd].plot(state_mn[:,3*mm+dd]+2*state_sd[:,3*mm+dd], ':',
                                                         color=color_list[mm])
        axs[dd].plot(state_mn[:,3*mm+dd]-2*state_sd[:,3*mm+dd], ':',
                                                         color=color_list[mm])

filename = "./results/mocap-mcmc-{}.p".format(model_type)
learner.save(filename)
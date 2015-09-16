import pickle
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt

from kalman import GaussianDensity
from linear_models import BasicLinearModel, DegenerateLinearModel
from learners_mcmc import (
    load_learner,
    BaseMCMCLearner,
    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
    MCMCLearnerTransitionBasicModelWithMNIWPrior,
    MCMCLearnerTransitionDegenerateModelWithMNIWPrior)

# Create learner classes
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

def mocap_msvd(markers, initial, prop_energy=0.95, num_it=1000):
    md = np.isnan(markers)
    markers = np.where(md, initial, markers)
    last_S = np.nan*np.ones(markers.shape[1])

    for it in range(num_it):

        U,S,V = la.svd(markers)
        if np.allclose(S,last_S,1E-7):
            break

        # Choose number of singular values
        normS = S/np.sum(S)
        cumsumS = np.cumsum(normS)
        num_sing = np.where( cumsumS>prop_energy )[0][0]

        # Reconstruct and fill in gaps
        reconstructed = np.dot(U[:,:num_sing], np.dot(np.diag(S[:num_sing]),
                                                              V[:num_sing,:]))
        markers = np.where(md, reconstructed, markers)

        # Keep track of singular values in order to assess convergence
        last_S = S

    return markers

def mocap_rmse(truth, original, estimate):
    err = truth-estimate
    return la.norm(err[np.isnan(original)])

# Load the test data
data_path = './mocap-data/'
test_path = './results/'#'./results/N20000/'#'./results/N2000/'#
markers_truth = np.genfromtxt(data_path+'downsampled_head_markers_truth.csv', delimiter=',')
test_data_file = 'mocap-test-data.p'
fh = open(test_path+test_data_file, 'rb')
markers = pickle.load(fh)

# Load the MCMC output
naive_learner = load_learner(test_path+'mocap-mcmc-naive.p')
basic_learner = load_learner(test_path+'mocap-mcmc-basic.p')
degenerate_learner = load_learner(test_path+'mocap-mcmc-degenerate.p')
#degenerate_learner = load_learner('longresults/intermediate-results-60000.p')

# Get state estimates from each algorithm
num_burn = 10000
basic_mn, basic_sd = basic_learner.estimate_state_trajectory(
                                                           numBurnIn=num_burn)
degenerate_mn, degenerate_sd = degenerate_learner.estimate_state_trajectory(
                                                           numBurnIn=num_burn)
naive_mn, naive_sd = naive_learner.estimate_state_trajectory(
                                                           numBurnIn=num_burn)

# Run MSVD as a comparison
msvd_markers = mocap_msvd(markers, naive_mn[:,:12])

# Assess RMSE
basic_rmse = mocap_rmse(markers_truth, markers, basic_mn[:,:12])
degenerate_rmse = mocap_rmse(markers_truth, markers, degenerate_mn[:,:12])
naive_rmse = mocap_rmse(markers_truth, markers, naive_mn[:,:12])
msvd_rmse = mocap_rmse(markers_truth, markers, msvd_markers)

# Display results
print("Model          | RMSE")
print("NCVM           | {}".format(naive_rmse))
print("Basic          | {}".format(basic_rmse))
print("Degenerate     | {}".format(degenerate_rmse))
print("MSVD           | {}".format(msvd_rmse))

# Trace plots
naive_learner.plot_chain_trace('R', dims=([0],[0]))
basic_learner.plot_chain_trace('R', dims=([0],[0]))
degenerate_learner.plot_chain_trace('R', dims=([0],[0]))
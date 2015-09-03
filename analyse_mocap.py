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

def mocap_msvd(init_markers, prop_energy=0.95, num_it=1000):
    markers = init_markers.copy()
    markers[np.where(np.isnan(init_markers))] = 0
    last_S = np.nan*np.ones(init_markers.shape[1])
    
    for it in range(num_it):
        
        U,S,V = la.svd(markers)
        if np.allclose(S,last_S,1E-7):
            break

        # Choose number of singular values
        normS = S/np.sum(S)
        cumsumS = np.cumsum(normS)
        num_sing = np.where( cumsumS>prop_energy )[0][0]
        
        # Reduce components
        U = U[:,:num_sing]
        V = V[:num_sing,:]
        Smat = np.diag(S[:num_sing])
        
        # Reconstruct
        reconstructed = np.dot(U,np.dot(Smat,V))
        markers[np.where(np.isnan(init_markers))] \
                             = reconstructed[np.where(np.isnan(init_markers))]

        last_S = S
        
    return markers

# Load the test data
test_data_file = './results/mocap-test-data.p'
fh = open(test_data_file, 'rb')
markers = pickle.load(fh)

# Load the MCMC output
naive_learner = load_learner('mcmc-mocap-naive')
basic_learner = load_learner('mcmc-mocap-basic')
degenerate_learner = load_learner('mcmc-mocap-degenerate')

# Run MSVD as a comparison
msvd_markers = mocap_msvd(markers)

# Get state estimates from each algorithm
state_mn, state_sd = learner.estimate_state_trajectory(numBurnIn=num_burn)

# Asses RMSE
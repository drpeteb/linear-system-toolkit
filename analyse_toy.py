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
class MCMCLearnerB(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionBasicModelWithMNIWPrior):
    pass

class MCMCLearnerD(BaseMCMCLearner,
                    MCMCLearnerObservationDiagonalCovarianceWithIGPrior,
                    MCMCLearnerTransitionDegenerateModelWithMNIWPrior):
    pass

def rmse(state, estimate):
    return la.norm(state-estimate)

figure_path = './notes/figures/'

# Load the test data
test_path = './results/toy-final/'
test_data_file = 'toy-test-data.p'
fh = open(test_path+test_data_file, 'rb')
model,state,observ = pickle.load(fh)

# Load the MCMC output
basic_learner = load_learner(test_path+'toy-mcmc-basic.p')
degenerate_learner = load_learner(test_path+'toy-mcmc-degenerate.p')

# Get state estimates from each algorithm
num_burn = 5000
basic_mn, basic_sd = basic_learner.estimate_state_trajectory(
                                                           numBurnIn=num_burn)
degenerate_mn, degenerate_sd = degenerate_learner.estimate_state_trajectory(
                                                           numBurnIn=num_burn)

# Assess RMSE
basic_rmse = rmse(state, basic_mn)
degenerate_rmse = rmse(state, degenerate_mn)

# Display results
print("Model          | RMSE")
print("Basic          | {}".format(basic_rmse))
print("Degenerate     | {}".format(degenerate_rmse))



# Draw state
ds = basic_learner.model.ds
fig, axs = plt.subplots(nrows=ds,ncols=1)
for dd in range(ds):
    axs[dd].plot(state[:,dd], 'k')
fig.savefig(figure_path+'toy-state.pdf', bbox_inches='tight')

# Trace and acf plots
basic_learner.plot_chain_trace('R', dims=([0],[0]), trueModel=model)
plt.gcf().axes[0].set_ylim([0.05,0.15])
plt.gcf().savefig(figure_path+'toy-basic-R-trace.pdf', bbox_inches='tight')
degenerate_learner.plot_chain_trace('R', dims=([0],[0]), trueModel=model)
plt.gcf().axes[0].set_ylim([0.05,0.15])
plt.gcf().savefig(figure_path+'toy-degenerate-R-trace.pdf', bbox_inches='tight')
basic_learner.plot_chain_acf('R', dims=([0],[0]), nlags=8)
plt.gcf().savefig(figure_path+'toy-basic-R-acf.pdf', bbox_inches='tight')
degenerate_learner.plot_chain_acf('R', dims=([0],[0]), nlags=8)
plt.gcf().savefig(figure_path+'toy-degenerate-R-acf.pdf', bbox_inches='tight')

# Histograms
degen_F_samples = np.array([pp['F'] for pp in degenerate_learner.chain_model])
Fmin = np.zeros((ds,ds))
Fmax = np.zeros((ds,ds))
for ii in range(ds):
    for jj in range(ds):
        Fmin[ii,jj] = np.round(np.min(degen_F_samples[:,ii,jj]),2)
        Fmax[ii,jj] = np.round(np.max(degen_F_samples[:,ii,jj]),2)
degenerate_learner.plot_chain_histogram('F', trueModel=model)
plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
for ii in range(ds):
    for jj in range(ds):
        aa = ii*ds+jj
        ax = plt.gcf().axes[aa]
        ax.set_yticklabels([])
        ax.set_xticks([Fmin[ii,jj],Fmax[ii,jj]])
plt.gcf().savefig(figure_path+'toy-degenerate-F-hist.pdf', bbox_inches='tight')

degen_Q_samples = np.array([np.dot(pp['vec'], np.dot(np.diag(pp['val']),pp['vec'].T)) for pp in degenerate_learner.chain_model])
fig, axs, coords =  degenerate_learner._create_2d_plot_axes(model.parameters['Q'])
for idx in np.ndindex(coords.shape):
    samples = [pp[coords[idx]] for pp in degen_Q_samples[num_burn:]]
    axs[idx].hist(samples, color='0.8')
    ylims = axs[idx].get_ylim()
    axs[idx].plot([model.parameters['Q'][coords[idx]]]*2, ylims, 'r',
                                                          linewidth=2)
    axs[idx].set_ylim(ylims)
Qmin = np.zeros((ds,ds))
Qmax = np.zeros((ds,ds))
for ii in range(ds):
    for jj in range(ds):
        Qmin[ii,jj] = np.round(np.min(degen_Q_samples[:,ii,jj]),2)
        Qmax[ii,jj] = np.round(np.max(degen_Q_samples[:,ii,jj]),2)
plt.gcf().subplots_adjust(wspace=0.3, hspace=0.3)
for ii in range(ds):
    for jj in range(ds):
        aa = ii*ds+jj
        ax = plt.gcf().axes[aa]
        ax.set_yticklabels([])
        ax.set_xticks([Qmin[ii,jj],Qmax[ii,jj]])
plt.gcf().savefig(figure_path+'toy-degenerate-Q-hist.pdf', bbox_inches='tight')
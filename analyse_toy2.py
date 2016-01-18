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
test_path = './results/toy2-final/'
test_data_file = 'toy2-test-data.p'
fh = open(test_path+test_data_file, 'rb')
model,state,observ = pickle.load(fh)

# Load the MCMC output
degenerate_learner = load_learner(test_path+'toy2-mcmc-degenerate.p')

# Get state estimates from each algorithm
num_burn = 5000
degenerate_mn, degenerate_sd = degenerate_learner.estimate_state_trajectory(
                                                           numBurnIn=num_burn)

# Draw state
ds = degenerate_learner.model.ds
fig, axs = plt.subplots(nrows=ds,ncols=1,sharex=True)
for dd in range(ds):
    axs[dd].plot(state[:,dd], 'k')
    axs[dd].locator_params(axis='y',nbins=2)
fig.savefig(figure_path+'toy2-state.pdf', bbox_inches='tight')

# Rank plots
degenerate_learner.plot_chain_trace('rank', numBurnIn=num_burn, trueModel=model)
plt.gcf().set_size_inches([8,3])
plt.gcf().axes[0].set_ylim([0,6])
plt.gcf().savefig(figure_path+'toy2-degenerate-rank-trace.pdf', bbox_inches='tight')

chain_rank = [mm['rank'][0] for mm in degenerate_learner.chain_model]
bins = np.array(range(0,np.max(chain_rank)+1))-0.5
rank_hist = np.histogram(chain_rank[num_burn:], bins=bins)
wid = 0.8;
f = plt.figure(figsize=[8,3])
ax = f.add_subplot(1,1,1)
ax.bar(rank_hist[1][:-1]+(0.5-wid/2),rank_hist[0],width=wid,color='0.5')
ax.plot([model.parameters['rank'][0]]*2, [0,len(chain_rank[num_burn:])], '-r')
ax.set_xlim([-0.5,np.max(chain_rank)+0.5])
ax.set_ylim([0,len(chain_rank[num_burn:])])
ax.locator_params(axis='x',nbins=len(bins)+1)
f.savefig(figure_path+'toy2-degenerate-rank-hist.pdf', bbox_inches='tight')

#degenerate_learner.plot_chain_histogram('rank', numBurnIn=num_burn, trueModel=model)
#plt.gcf().savefig(figure_path+'toy2-degenerate-rank-hist.pdf', bbox_inches='tight')







# Operational modules
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.linalg as la
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt

# Import from other module files
import sampling as smp
from linear_models import BasicLinearModel

class AbstractMCMCLearner:
    __metaclass__ = ABCMeta
    """
    Abstract Container class for MCMC system learning algorithms.
    Sub-class this and implement methods to perform individual MCMC sampling 
    steps. These will depend on the type of linear model and the choice of
    prior distribusion.
    """
    
    def __init__(self, initial_model_estimate, observ, hyperparams,
                                            algoparams=dict(), verbose=False):
        self.model = initial_model_estimate
        self.observ = observ
        self.hyperparams = hyperparams
        self.algoparams = algoparams
        
        self.chain = []
        self.chain_model = []
        self.chain_state = []
        self.chain_lhood = []
        
        self.verbose = verbose
    
    def save_link(self):
        """Save the current state of the model as a link in the chain"""
        self.chain.append(self.model.copy())
    
    
    def _create_2d_plot_axes(self, paramName, index1=None, index2=None):
        """
        Create a figure, axes, and coordinates for elements of a 2D array
        """
        
        # Get the parameter shape
        paramShape = self.model.parameters[paramName].shape
        
        # Create index lists if missing
        if index1 is None:
            numRows = paramShape[0]
            index1 = list(range(numRows))
        else:
            numRows = len(index1)
        if index2 is None:
            numCols = paramShape[1]
            index2 = list(range(numCols))
        else:
            numCols = len(index2)
        
        # Create figure and axes
        fig, axs = plt.subplots(nrows=numRows, ncols=numCols, squeeze=False)
        
        # Make an array of tuples indexing the right element of the parameter
        # for each subplot
        coords = np.empty((numRows,numCols),dtype=tuple)
        for rr in range(numRows):
            for cc in range(numCols):
                coords[rr,cc] = (index1[rr],index2[cc])
        
        return fig, axs, coords
    
    def _create_1d_plot_axes(self, paramName, index=None):
        """
        Create a figure, axes, and coordinates for elements of a 1D array
        """
        
        # Get the parameter shape
        paramShape = self.model.parameters[paramName].shape
        
        # Create index lists if missing
        if index is None:
            numEls = paramShape[0]
            index = list(range(numEls))
        else:
            numEls = len(index)
        
        # Create figure and axes
        fig, axs = plt.subplots(nrows=1, ncols=numEls, squeeze=False)
        axs = axs.reshape((numEls,))
        
        # Make an array indexing the right element of the parameter
        # for each subplot
        coords = np.empty((numEls),dtype=tuple)
        for ee in range(numEls):
            coords[ee] = (index[ee],)
        
        return fig, axs, coords
    
    
    def plot_chain_trace(self, paramName, numBurnIn=0, dims=None):
        """
        Make Markov chain trace plots for a chosen parameter
        
        dims is a list or tuple of two lists which specificy which rows and 
        columns should be plotted. If empty then all are plotted.
        """
        
        # Get the parameter shape
        paramShape = self.model.parameters[paramName].shape
        
        if len(paramShape)==1:
            fig, axs, coords = self._create_1d_plot_axes(paramName, dims)
        elif len(paramShape)==2:
            if dims is None:
                dims = (None,None)
            fig, axs, coords = self._create_2d_plot_axes(paramName, dims[0],
                                                                      dims[1])
        else:
            raise ValueError("Cannot draw plots for this parameter")
        
        for idx in np.ndindex(coords.shape):
            samples = [mod.parameters[paramName][coords[idx]] \
                                                        for mod in self.chain]
            axs[idx].plot(samples, 'k')
            ylims = axs[idx].get_ylim()
            axs[idx].plot([numBurnIn]*2, ylims, 'r:')
            axs[idx].set_ylim(ylims)
    
    
    def plot_chain_histogram(self, paramName, numBurnIn=0, dims=None,
                                                              trueValue=None):
        """
        Make Markov chain histograms for a chosen parameter
        
        dims is a tuple of two lists specificy which rows and columns should
        be plotted. If empty then all are plotted.
        """
        
        # Get the parameter shape
        paramShape = self.model.parameters[paramName].shape
        
        if len(paramShape)==1:
            fig, axs, coords = self._create_1d_plot_axes(paramName, dims)
        elif len(paramShape)==2:
            if dims is None:
                dims = (None,None)
            fig, axs, coords = self._create_2d_plot_axes(paramName, dims[0],
                                                                      dims[1])
        else:
            raise ValueError("Cannot draw plots for this parameter")
        
        for idx in np.ndindex(coords.shape):
            samples = [mod.parameters[paramName][coords[idx]] \
                                            for mod in self.chain[numBurnIn:]]
            axs[idx].hist(samples, color='0.8')
            if trueValue is not None:
                ylims = axs[idx].get_ylim()
                axs[idx].plot([trueValue[coords[idx]]]*2, ylims, 'r',
                                                                  linewidth=2)
                axs[idx].set_ylim(ylims)





class MCMCLearnerForBasicModelWithMNIWPrior(AbstractMCMCLearner):
    """
    Container for MCMC system learning algorithm.
    Model Type: Basic
    Prior Type: Matrix Normal-Inverse Wishart
    """
        
    def iterate_transition(self):
        """
        MCMC iteration (Gibbs sampling) for transition matrix and covariance
        """
        
        # First sample the state sequence
        x = self.model.sample_posterior(self.observ)
        
        # Calculate sufficient statistics
        suffStats = smp.evaluate_transition_sufficient_statistics(x)
        
        # Update hyperparameters
        nu,Psi,M,V = smp.hyperparam_update_basic_mniw_transition(
                                                    suffStats,
                                                    self.hyperparams['nu0'],
                                                    self.hyperparams['Psi0'],
                                                    self.hyperparams['M0'],
                                                    self.hyperparams['V0'])
        
        # Sample new parameters
        Q = la.inv(smp.sample_wishart(nu, la.inv(Psi)))
        F = smp.sample_matrix_normal(M, Q, V)
        
        # Update the model
        self.model.parameters['F'] = F
        self.model.parameters['Q'] = Q
    
    def iterate_observation_diagonal_covariance(self):
        """
        MCMC iteration (Gibbs sampling) for diagonal observation covariance
        matrix with inverse-gamma prior
        """
        
        # First sample the state sequence
        x = self.model.sample_posterior(self.observ)
        
        # Calculate sufficient statistics
        suffStats = smp.evaluate_observation_sufficient_statistics(x,
                                                                  self.observ)
        
        # Update hyperparameters
        a,b = smp.hyperparam_update_basic_ig_observation_variance(
                                    suffStats,
                                    self.model.parameters['H'],
                                    self.hyperparams['a0'],
                                    self.hyperparams['b0'])
        
        # Sample new parameter
        r = stats.invgamma.rvs(a, scale=b)
        self.model.parameters['R'] = r*np.identity(self.model.do)
        
        

#    def iterate_transition_matrix(self):
#        """
#        MCMC iteration (Gibbs sampling) for transition matrix only
#        """
#        
#        # First sample the state sequence
#        x = self.model.sample_posterior(self.observ)
#        
#        # Calculate sufficient statistics
#        suffStats = smp.evaluate_transition_sufficient_statistics(x)
#        
#        # Update hyperparameters
#        M,V = smp.hyperparam_update_basic_mniw_transition_matrix(
#                                                    suffStats,
#                                                    self.hyperparams['M0'],
#                                                    self.hyperparams['V0'])
#        
#        # Sample new parameters
#        Q = self.model.parameters['Q']
#        F = smp.sample_matrix_normal(M, Q, V)
#        
#        # Update the model
#        self.model.parameters['F'] = F
#
#    def iterate_transition_covariance(self):
#        """
#        MCMC iteration (Gibbs sampling) for transition covariance only
#        """
#        
#        # First sample the state sequence
#        x = self.model.sample_posterior(self.observ)
#        
#        # Calculate sufficient statistics
#        suffStats = smp.evaluate_transition_sufficient_statistics(x)
#        
#        # Update hyperparameters
#        nu,Psi = smp.hyperparam_update_basic_mniw_transition_covariance(
#                                                suffStats,
#                                                self.model.parameters['F'],
#                                                self.hyperparams['nu0'],
#                                                self.hyperparams['Psi0'])
#        
#        # Sample new parameters
#        Q = la.inv(smp.sample_wishart(nu, la.inv(Psi)))
#        
#        # Update the model
#        self.model.parameters['Q'] = Q
        
        
        
        
        
                      

class MCMCLearnerForDegenerateModelWithMNIWPrior(MCMCLearnerForBasicModelWithMNIWPrior):
    """
    Container for MCMC system learning algorithm.
    Model Type: Degenerate
    Prior Type: Singular Matrix Normal-Inverse Wishart
    """
    
    def transition_prior(self, model):
        """
        Prior density for transition model parameters
        """
        variancePrior = smp.singular_inverse_wishart_density(
                                            model.parameters['val'],
                                            model.parameters['vec'],
                                            la.inv(self.hyperparams['Psi0']))
        
        orthVec = model.complete_basis()
#        relaxEval = self.hyperparams['alpha']
        relaxEval = np.min(model.parameters['val'])
        rowVariance = model.transition_covariance() \
                      + relaxEval*np.dot(orthVec,orthVec.T)
        matrixPrior = smp.matrix_normal_density(model.parameters['F'],
                                                self.hyperparams['M0'],
                                                rowVariance,
                                                self.hyperparams['V0'])
        
        return variancePrior + matrixPrior
    
    def iterate_transition(self, moveType):
        """
        MCMC iteration (Metropolis-Hastings) for transition matrix and
        covariance.
        """
        
        # Kalman filter
        flt,_,lhood = self.model.kalman_filter(self.observ)
        
        # Copy model
        ppsl_model = self.model.copy()
        
        if self.verbose:
            print("Metropolis-Hastings move type: {}".format(moveType))
        
        # Propose change
        if   moveType=='F':
            
            # Simulate state trajectory
            x = ppsl_model.backward_simulation(flt)
            suffStats = smp.evaluate_transition_sufficient_statistics(x)
            
            # Pad the transition covariance
            padded_Q = ppsl_model.transition_covariance() + \
                              self.algoparams['Fs']*np.identity(ppsl_model.ds)
            
            # Sample a new transition matrix
            M,V = smp.hyperparam_update_basic_mniw_transition_matrix(
                                                    suffStats,
                                                    self.hyperparams['M0'],
                                                    self.hyperparams['V0'],)
            ppsl_F = smp.sample_matrix_normal(M,padded_Q,V)
            fwd_prob = smp.matrix_normal_density(ppsl_F,M,padded_Q,V)
            ppsl_model.parameters['F'] = ppsl_F
            
            # Sample a new trajectory
            ppsl_x = ppsl_model.sample_posterior(self.observ)
            ppsl_suffStats = smp.evaluate_transition_sufficient_statistics(
                                                                       ppsl_x)
            
            # Reverse move probaility
            M,V = smp.hyperparam_update_basic_mniw_transition_matrix(
                                                    ppsl_suffStats,
                                                    self.hyperparams['M0'],
                                                    self.hyperparams['V0'],)
            bwd_prob = smp.matrix_normal_density(self.model.parameters['F'],
                                                                 M,padded_Q,V)
        
        elif moveType=='Q':
            
            # Make the change
            rotation = smp.sample_cayley(self.model.ds, self.algoparams['Qs'])
            ppsl_model.rotate_transition_covariance(rotation)
            
            # Random walk, so forward and backward probabilities are same
            fwd_prob = 0
            bwd_prob = 0
            
        elif moveType=='rank':
            
            if ppsl_model.ds == 1:
                switch = -1                               # Do nothing (1D)
            elif ppsl_model.parameters['rank'][0] == 1:
                switch = 1                                # Must increase rank
            elif ppsl_model.parameters['rank'][0] == ppsl_model.ds:
                switch = 0                                # Must decrease rank
            else:
                switch = np.random.random_integers(0,1)   # choose at random
            
            if switch == -1:
                # No options to change rank
                
                prior = 0
                ppsl_prior = 0
                fwd_prob = 0
                bwd_prob = 0
                
            elif switch == 0:
                # Decrease rank
            
                if self.verbose:
                    print("   Decreasing covariance rank")
        
                
                # Remove the smallest eigenvalue and associated eigenvector
                oldValue, oldVector = \
                                    ppsl_model.remove_min_eigen_value_vector()
                
                # Reverse move prob for adding the smallest e-value
                minValue = np.min(ppsl_model.parameters['val'])
                valPpslProb = -np.log( minValue )
                
                # Nullspace
                nullSpace = ppsl_model.complete_basis()
                nullDims = nullSpace.shape[1]
                
                # Calculate the Jacobian
                constJac = 0.5*nullDims*np.log(np.pi) \
                         - np.sum(np.log(ppsl_model.parameters['val'])) \
                         - special.gammaln(nullDims/2)
                varJac = nullDims*np.log(oldValue) \
                       + np.sum(np.log(ppsl_model.parameters['val']-oldValue))
                
                # Fudge the proposals and jacobian into the proposal terms
                fwd_prob = constJac + varJac
                bwd_prob = valPpslProb
                
            elif switch == 1:
                # Increase rank
            
                if self.verbose:
                    print("   Increasing covariance rank")
                
                # Sample a new eigenvalue between 0 and the smallest e-value
                minValue = np.min(ppsl_model.parameters['val'])
                newValue = stats.uniform.rvs(loc=0, scale=minValue)
                valPpslProb = -np.log( minValue )
                
                # Sample a new eigenvector
                nullSpace = ppsl_model.complete_basis()
                nullDims = nullSpace.shape[1]
                coefs = smp.sample_orthogonal_haar(nullDims)
                newVector = np.dot(nullSpace, coefs)
                
                # Calculate the Jacobian
                constJac = 0.5*nullDims*np.log(np.pi) \
                         - np.sum(np.log(ppsl_model.parameters['val'])) \
                         - np.log(2.0) \
                         - special.gammaln(nullDims/2)
                varJac = nullDims*np.log(newValue) \
                       + np.sum(np.log(ppsl_model.parameters['val']-newValue))
                                # Add them to the model
                ppsl_model.add_eigen_value_vector(newValue, newVector)
                
                # Fudge the proposals and jacobian into the proposal terms
                fwd_prob = valPpslProb
                bwd_prob = constJac + varJac
                
                # Assumes uniform prior on each possible value of the rank
                
        else:
            raise ValueError("Invalid move type")
        
        # Kalman filter
        ppsl_flt,_,ppsl_lhood = ppsl_model.kalman_filter(self.observ)
        
        # Prior terms
        prior = self.transition_prior(self.model)
        ppsl_prior = self.transition_prior(ppsl_model)
        
        # Decide
        acceptRatio =   (ppsl_lhood-lhood) \
                      + (ppsl_prior-prior) \
                      + (bwd_prob-fwd_prob)
        if self.verbose:
                print("   Acceptance ratio: {}".format(acceptRatio))
        if np.log(np.random.random())<acceptRatio:
            self.model = ppsl_model
            flt = ppsl_flt
            if self.verbose:
                print("   accepted")
        else:
            if self.verbose:
                print("   rejected")
        
        # Sample within subspace
        self.iterate_transition_within_subspace(flt)
    
    
    def iterate_transition_within_subspace(self, flt):
        """
        MCMC iteration (Gibbs sampling) for transition matrix and covariance
        within the constrained subspace
        """
        
        # First sample the state sequence
        x = self.model.backward_simulation(flt)
        
        # Calculate sufficient statistics
        suffStats = smp.evaluate_transition_sufficient_statistics(x)
        
        # Convert to Givens factorisation form
        U,D = self.model.convert_to_givens_form()
        
        # Sample a new projected transition matrix and transition covariance
        nu,Psi,M,V = smp.hyperparam_update_degenerate_mniw_transition(
                                                    suffStats, U,
                                                    self.hyperparams['nu0'],
                                                    self.hyperparams['Psi0'],
                                                    self.hyperparams['M0'],
                                                    self.hyperparams['V0'])
        D = la.inv(smp.sample_wishart(nu, la.inv(Psi)))
        FU = smp.sample_matrix_normal(M, D, V)
        
        # Project out
        Fold = self.model.parameters['F']
        F = smp.project_degenerate_transition_matrix(Fold, FU, U)
        self.model.parameters['F'] = F
        
        # Convert back to eigen-decomposition form
        self.model.update_from_givens_form(U, D)
    
    
#            
#        
#    def save_link(self):
#        """Save the current state of the model as a link in the chain"""
#        self.chain.append(self.model.copy())
#        if self.model_type==2:
#            self.pad_std_chain.append(self.pad_std)
#            self.rot_std_chain.append(self.rot_std)
#            self.giv_std_chain.append(self.giv_std)
##            for gg in range(len(self.model.givens)):
##                self.giv_std_chain[gg].append(self.giv_std[gg])
#                
#    def adapt_std(self, B):
#        """Adapt proposal distribution parameters"""
#        ar = np.mean(np.array(self.F_acc[-B:]))
#        bn = len(self.F_acc)/B
#        delta = min(0.1,1/np.sqrt(bn))
#        pad_min = 1E-6
#        if (ar<0.44):
#            self.pad_std = np.maximum(pad_min, self.pad_std*np.exp(-delta))
#        else:
#            self.pad_std = np.maximum(pad_min, self.pad_std*np.exp(delta))
#            
#        ar = np.mean(np.array(self.rot_acc[-B:]))
#        bn = len(self.rot_acc)/B
#        delta = min(0.1,1/np.sqrt(bn))
#        rot_min = 1E-4
#        if (ar<0.44):
#            self.rot_std = np.maximum(rot_min, self.rot_std*np.exp(-delta))
#        else:
#            self.rot_std = np.maximum(rot_min, self.rot_std*np.exp(delta))
#        
#        ar = np.mean(np.array(self.givens_acc[-B:]))
#        bn = len(self.givens_acc)/B
#        delta = min(0.1,1/np.sqrt(bn))
#        if (ar<0.44):
#            self.giv_std = self.giv_std*np.exp(-delta)
#        else:
#            self.giv_std = self.giv_std*np.exp(delta)
##        
##        for gg in range(len(self.model.givens)):
##            ar = np.mean(np.array(self.givens_acc[gg][-B:]))
##            bn = len(self.givens_acc[gg])/B
##            delta = min(0.1,1/np.sqrt(bn))
##            if (ar<0.44):
##                self.giv_std[gg] = self.giv_std[gg]*np.exp(-delta)
##            else:
##                self.giv_std[gg] = self.giv_std[gg]*np.exp(delta)

            
#### DISPLAY ###

#    
#
#def draw_chain_autocorrelation(chain, true_model, burn_in, fields, nlags=30):
#    d = min(4,true_model.ds) # Don't plot too many components - too small
#    
#    if len(chain)>burn_in:
#        
#        for ff in fields:
#            if ff in ['F','Q','G','D']:
#                d = getattr(true_model, ff).shape[0]
#                d = min(4,d)
#                fig,ax = plt.subplots(d,d,squeeze=False,figsize=(9,9))
#                fig.subplots_adjust(wspace=0.3, hspace=0.3)
#                for rr in range(d):
#                    for cc in range( getattr(chain[-1],ff).shape[1] ):
#                        seq = [getattr(mm,ff)[rr,cc] for mm in chain[burn_in:]]
#                        ac = st.acf(seq,unbiased=False,nlags=nlags,)
#                        ax[rr,cc].locator_params(nbins=2)
#                        ax[rr,cc].set_xlim((0,nlags))
#                        ax[rr,cc].set_ylim((-0.1,1))
#                        ax[rr,cc].plot(range(len(ac)),ac,'k')
#                        ax[rr,cc].plot(range(len(ac)),[0]*len(ac),':k')
#            elif ff in ['givens']:
#                d = len(getattr(true_model,ff))
#                fig = plt.figure()
#                for dd in range(d):
#                    ax = fig.add_subplot(d,1,dd+1)
#                    seq = [getattr(mm,ff)[dd] for mm in chain[burn_in:]]
#                    ac = st.acf(seq,unbiased=False,nlags=nlags,)
#                    ax.plot(range(len(ac)),ac,'k')
#                    ax.plot(range(len(ac)),[0]*len(ac),':k')
#                    ax.set_ylim([-0.1,1])
#                    
#    return fig
#
#
#def draw_chain_acceptance(chain_acc, burn_in):
#    d = len(chain_acc)
#    ax = plt.figure().add_subplot(1,1,1)
#    maxval = 0
#    for dd in range(d):
#        acc_array = np.array(chain_acc[dd])
#        ax.plot(np.cumsum(acc_array))
#        maxval = max(maxval,acc_array.sum())
#    ax.plot([burn_in]*2,[0,maxval])
#
#
#def draw_chain_adaptation(chain_adpt, burn_in):
#    d = len(chain_adpt)
#    ax = plt.figure().add_subplot(1,1,1)
#    maxval = 0
#    for dd in range(d):
#        adpt_array = np.array(chain_adpt[dd])
#        ax.plot(adpt_array)
#        maxval = max(maxval,adpt_array.max())
#    ax.plot([burn_in]*2,[0,maxval])
#
#     
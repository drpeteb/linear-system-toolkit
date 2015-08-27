# Operational modules
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.linalg as la
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
            samples = [mod.parameters[paramName][coords[idx]] for mod in self.chain]
            axs[idx].plot(samples, 'k')
    
    
    def plot_chain_histogram(self, paramName, numBurnIn=0, dims=None, trueValue=None):
        """
        Make Markov chain histograms for a chosen parameter
        
        dims is a tuple of two lists specificy which rows and columns should be
        plotted. If empty then all are plotted.
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
            samples = [mod.parameters[paramName][coords[idx]] for mod in self.chain]
            axs[idx].hist(samples, color='0.8')
            if trueValue is not None:
                ylims = axs[idx].get_ylim()
                axs[idx].plot([trueValue[coords[idx]]]*2, ylims, 'r', linewidth=2)
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
        suffStats = smp.evaluate_sufficient_statistics(x)
        
        # Sample a new transition matrix and transition covariance
        self.model.parameters['F'], self.model.parameters['Q'] = \
             smp.sample_basic_transition_mniw_conditional(suffStats,
                                                    self.hyperparams['nu0'],
                                                    self.hyperparams['Psi0'],
                                                    self.hyperparams['M0'],
                                                    self.hyperparams['V0'])
                                                    

class MCMCLearnerForDegenerateModelWithMNIWPrior(AbstractMCMCLearner):
    """
    Container for MCMC system learning algorithm.
    Model Type: Degenerate
    Prior Type: Singular Matrix Normal-Inverse Wishart
    """
    
    def transition_prior(self, model):
        """
        Prior density for transition model parameters
        """
        variancePrior = smp.singular_wishart_density(model.parameters['val'],
                      model.parameters['vec'],la.inv(self.hyperparams['Psi0']))
        
        orthVec = model.complete_basis()
        rowVariance = model.transition_covariance() \
                      + self.hyperparams['alpha']*np.dot(orthVec,orthVec.T)
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
        
        # Propose change
        if   moveType=='F':
            
            # Simulate state trajectory
            x = ppsl_model.backward_simulation(flt)
            suffStats = smp.evaluate_sufficient_statistics(x)
            
            # Pad the transition covariance
            padded_Q = ppsl_model.transition_covariance() + \
                               self.algoparams['Fs']*np.identity(ppsl_model.ds)
            
            # Sample a new transition matrix
            ppsl_F,fwd_prob = smp.sample_basic_transition_matrix_mniw_conditional(
                                            suffStats,
                                            padded_Q,
                                            self.hyperparams['M0'],
                                            self.hyperparams['V0'],
                                            with_pdf=True)
            ppsl_model.parameters['F'] = ppsl_F
            
            # Sample a new trajectory
            ppsl_x = ppsl_model.sample_posterior(self.observ)
            ppsl_suffStats = smp.evaluate_sufficient_statistics(ppsl_x)
            
            # Reverse move probaility
            _,bwd_prob = smp.sample_basic_transition_matrix_mniw_conditional(
                                            ppsl_suffStats,
                                            padded_Q,
                                            self.hyperparams['M0'],
                                            self.hyperparams['V0'],
                                            F = self.model.parameters['F'],
                                            with_pdf=True)
            ppsl_model.parameters['F'] = ppsl_F
            
            # Prior terms
            prior = self.transition_prior(self.model)
            ppsl_prior = self.transition_prior(ppsl_model)
        
        elif moveType=='Q':
            
            # Make the change
            rotation = smp.sample_cayley(self.model.ds, self.algoparams['Qs'])
            ppsl_model.rotate_transition_covariance(rotation)
            
            # Random walk, so forward and backward probabilities are same
            fwd_prob = 0
            bwd_prob = 0
            
            # Prior terms
            prior = self.transition_prior(self.model)
            ppsl_prior = self.transition_prior(ppsl_model)
            
        elif moveType=='rank':
            pass
        else:
            raise ValueError("Invalid move type")
        
        if self.verbose:
            print("Metropolis-Hastings move type: {}".format(moveType))
        
        # Kalman filter
        ppsl_flt,_,ppsl_lhood = ppsl_model.kalman_filter(self.observ)
        
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
        suffStats = smp.evaluate_sufficient_statistics(x)
        
        # Convert to Givens factorisation form
        U,D = self.model.convert_to_givens_form()
        
        # Sample a new transition matrix and transition covariance
        self.model.parameters['F'], D = \
             smp.sample_degenerate_transition_mniw_conditional(
                                                    suffStats, U,
                                                    self.model.parameters['F'],
                                                    self.hyperparams['nu0'],
                                                    self.hyperparams['Psi0'],
                                                    self.hyperparams['M0'],
                                                    self.hyperparams['V0'])
        
        # Convert back to eigen-decomposition form
        self.model.update_from_givens_form(U, D)
    
    
    

#class MCMCLearnerForDegenerateModelWithIndependentPriors(AbstractMCMCLearner):
#    """
#    Container for MCMC system learning algorithm.
#    Model Type: Degenerate
#    Prior Type: Independent Matrix Normal and Singular Inverse Wishart
#    """
#        
#    def iterate_transition(self):
#        """
#        MCMC iteration (Gibbs sampling) for transition matrix and covariance
#        """
#        
#        # First sample the state sequence
#        x = self.model.sample_posterior(self.observ)
#        
#        # Convert to Givens factorisation form
#        U,D = self.model.convert_to_givens_form()
#        
#        # Sample a new transition matrix and transition covariance
#        self.model.parameters['F'], D = \
#             smp.sample_degenerate_transition_independent_conditional(x, U,
#                                                self.hyperparams['psi0'],
#                                                self.hyperparams['M0'],
#                                                self.hyperparams['alpha'],
#                                                F=self.model.parameters['F'])
#        
#        # Convert back to eigen-decomposition form
#        self.model.update_from_givens_form(U, D)





#import matplotlib.pyplot as plt
#import numpy as np
#import linear_models_sampling as sysl
#from scipy import linalg as la
#from scipy import stats
#from scipy.stats import multivariate_normal as mvn
#import statsmodels.tsa.stattools as st
#from linear_models import LinearModel, SparseLinearModel, DegenerateLinearModel
#
#class LinearModelMCMC:
#    """Container Class for Linear-Gaussian Model Learning with MCMC"""
#
#    def __init__(self, model_type, init_model_est, observ, A_prior_vr=None, B_prior_prob=None, Q_prior_scale=None, Q_prior_dof=None, giv_std=None, pad_std=None, rot_std=None):
#        self.model_type = model_type
#        
#        self.model = init_model_est.copy()
#        self.observ = observ
#
#        self.chain = []
#        self.chain.append(self.model.copy())        
#        
#        self.A_prior_vr = A_prior_vr
#        self.B_prior_prob = B_prior_prob
#        self.Q_prior_scale = Q_prior_scale # This is the scalar p such that the Q^{-1} is Wishart with mean nu*p*I (i.e. phi in the paper)
#        self.Q_prior_dof = Q_prior_dof
#        
#        if model_type==2:
#            
##            worst_rank = int(self.model.ds/2)
##            most_givens = worst_rank*self.model.ds-worst_rank
#            
##            self.giv_std = [giv_std]*(most_givens)
#            self.giv_std = giv_std
#            self.pad_std = pad_std
#            self.rot_std = rot_std
#            
#            self.F_acc = []
#            self.pad_std_chain = []
#            
#            self.rot_acc = []
#            self.rot_std_chain = []
#            
#            self.givens_acc = []
#            self.giv_std_chain = []
##            self.givens_acc = [[] for gg in range(most_givens)]
##            self.giv_std_chain = [[] for gg in range(most_givens)]
##            for gg in range(most_givens):
##                self.giv_std_chain[gg].append(self.giv_std[gg])
#                
#            self.rank_acc = []
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
#    
#    ### MCMC moves ###
#    
#    def iterate_transition_matrix(self, Mtimes):
#        """Run Gibbs iterations on transition matrix"""
#        
#        if (self.A_prior_vr is None):
#            raise ValueError("A transition matrix prior must be supplied in order to sample")
#        if (self.model_type==1) and (self.B_prior_prob is None):
#            raise ValueError("A mask prior must be supplied in order to sample")
#        
#        ds = self.model.ds
#        
#        for mm in range(Mtimes):
#        
#            # First sample the state sequence
#            x = self.model.sample_posterior(self.observ)
#            
#            if self.model_type == 0:
#            
#                # Sample a new transition matrix
#                self.model.F,_ = sysl.sample_transition_matrix_conditional(x, self.model.Q, self.A_prior_vr)
#                
#            elif self.model_type == 1:
#                
#                # Sample a new transition matrix (all elements jointly)
#                self.model.A,_ = sysl.sample_transition_matrix_conditional(x, self.model.Q, self.A_prior_vr, self.model.B)
#                
##                # Sample binary mask
##                self.model.B = sysl.sample_transition_matrix_mask_conditional(x, self.model.A, self.model.B, self.model.Q, self.B_prior_prob)
#            
#                # Sample transition matrix with binary mask (A,B pairs jointly)
#                self.model.A,self.model.B = sysl.sample_transition_matrix_and_mask_conditional(x, self.model.A, self.model.B, self.model.Q, self.A_prior_vr, self.B_prior_prob)
#                
#            elif self.model_type == 2:
#                
#                # Keep a copy of the current F
#                Fold = self.model.F.copy()
#                
##                try:
#                
#                # KF to get the likelihood
#                flt,_,old_lhood = self.model.kalman_filter(self.observ)
#                
#                # Sample a state trajectory
#                x = self.model.backward_simulation(flt)
#                
#                # Pad the degenerate covariance matrix
#                padding = (self.pad_std**2)*np.identity(ds)
#                Qpad = self.model.Q + padding
#                
#                # Sample a new transition matrix
#                Fppsl,F_fwd_ppsl_prob = sysl.sample_transition_matrix_conditional(x, Qpad, self.A_prior_vr)
#                self.model.F = Fppsl
#                
#                # New likelihood
#                ppsl_flt,_,new_lhood = self.model.kalman_filter(self.observ)            
#                
#                # Sample a new state sequence
#                xppsl = self.model.backward_simulation(ppsl_flt)
#                
#                # Reverse proposal
#                _,F_bwd_ppsl_prob = sysl.sample_transition_matrix_conditional(xppsl, Qpad, self.A_prior_vr, A=Fold)
#                
#                # Priors
#                new_prior = mvn.logpdf( Fppsl.flatten(), np.zeros((ds**2)), self.A_prior_vr*np.identity(ds**2) )
#                old_prior = mvn.logpdf( Fold.flatten(),  np.zeros((ds**2)), self.A_prior_vr*np.identity(ds**2) )
#                
#                # Accept?
#                ap = (new_lhood+new_prior) - (old_lhood+old_prior) + (F_bwd_ppsl_prob-F_fwd_ppsl_prob)
##                print(ap)
##                except:
##                    
##                    # If move fails (which very occasonally happens due to very nearly singular covariances) then just leave unchanged
##                    ap = -np.inf                
#
#                if np.log(np.random.random()) < ap:
#                    self.F_acc.append(1)
#                else:
#                    self.model.F = Fold
#                    self.F_acc.append(0)
#                
#                # Sample within subspace
#                if sum(self.F_acc)>1:
#                    x = self.model.sample_posterior(self.observ)
#                    F = sysl.sample_degenerate_transition_matrix_conditional(x, self.model.F, np.dot(self.model.D,self.model.D.T), self.model.U, self.A_prior_vr)
#                    self.model.F = F
#                
#            else:
#                raise ValueError("Invalid model type")
#    
#    def iterate_transition_noise_matrix(self, Mtimes):
#        """Run Gibbs iterations on the transition noise matrix"""
#        
#        if ((self.Q_prior_dof==None) or (self.Q_prior_scale==None)):
#            raise ValueError("A transition covariance matrix prior must be supplied in order to sample")
#            
#        for mm in range(Mtimes):
#            
#            if (self.model_type==0) or (self.model_type==1):
#                
#                # First sample the state sequence
#                x = self.model.sample_posterior(self.observ)
#                
#                # Sample a new covariance matrix
#                Q,_ = sysl.sample_transition_covariance_conditional(x, self.model.F, self.Q_prior_dof, self.Q_prior_scale)
#                
#                # Square root it
#                G = la.cholesky(Q,lower=True)
#                self.model.G = G
#                
#            elif self.model_type==2:
#
#                # Joint move
#                
#                # Old likelihood
#                flt,_,lhood = self.model.kalman_filter(self.observ)
#                
#                if self.model.rank < self.model.ds:
#                    
#                    # Make a proposal copy of the model
#                    ppsl_model = self.model.copy()
#                    
#                    # Sample a Cayley-distributed random matrix
#                    M = sysl.sample_cayley(ppsl_model.ds, self.rot_std)
#                    
#                    # Use it to rotate noise matrix
#                    ppsl_model.rotate_noise(M)
#                    
#                    # Likelihood
#                    new_flt,_,new_lhood = ppsl_model.kalman_filter(self.observ)
#                    
#                    # Accept probability
#                    ap = new_lhood-lhood
##                    print(ap)
#                    
#                    # Accept/Reject
#                    if np.log(np.random.random()) < ap:
#                        self.model = ppsl_model
#                        self.rot_acc.append(1)
#                        flt = new_flt
#                        lhood = new_lhood
#                    else:
#                        self.rot_acc.append(0)
#                    
#                    # How many planar moves should we do?
#                    num_givens_moves = ppsl_model.ds
#                    
#                    # Keep track of givens acceptances
#                    giv_acc = np.zeros(num_givens_moves)
#                    
#                    # Planar moves
#                    for gg in range(num_givens_moves):
#                        
#                        # Make a proposal copy of the model
#                        ppsl_model = self.model.copy()
#                        
#                        # Random plane
#                        ii = np.random.random_integers(ppsl_model.ds)-1
#                        jj = ii
#                        while jj == ii:
#                            jj = np.random.random_integers(ppsl_model.ds)-1
#                            
#                        # Random rotation
#                        rot = np.random.normal(0,self.giv_std)
#                        ppsl_model.planar_rotate_noise(ii, jj, rot)
#                        
#                        # Likelihood
#                        new_flt,_,new_lhood = ppsl_model.kalman_filter(self.observ)
#                    
#                        # Accept probability
#                        ap = new_lhood-lhood
##                        print(ap)
#                    
#                        # Accept/Reject
#                        if np.log(np.random.random()) < ap:
#                            self.model = ppsl_model
##                            self.givens_acc.append(1)
#                            giv_acc[gg] = 1
#                            flt = new_flt
#                            lhood = new_lhood
#                        else:
##                            self.givens_acc.append(0)
#                            giv_acc[gg] = 0
#                        
#                    self.givens_acc.append(np.mean(giv_acc))
#                
##                # Prior givens probability
##                old_givens_prior = self.model.givens_prior()
##                
##                # First sample the state sequence
##                flt,_,old_lhood = self.model.kalman_filter(self.observ)
##                x = self.model.backward_simulation(flt)
##                
##                if self.model.rank < self.model.ds:
##                    
##                    # Keep track of givens acceptances
##                    giv_acc = np.zeros(len(self.model.givens))
##                    
##                    for gg in range(len(self.model.givens)):
##                        
##                        # Store current
##                        #old_ga = self.model.givens[gg]
##                        ppsl_model = self.model.copy()
##                        
###                        print(self.model.order)
###                        print(ppsl_model.order)
##                        
##                        # Proposal standard deviation
###                        gstd = self.giv_std[gg]
##                        gstd = self.giv_std
##                        
##                        # Propose a random walk change and correct if it wraps
##                        new_ga = ppsl_model.givens[gg]+np.random.normal(0,gstd)
##                        wrap = ppsl_model.set_givens(gg, new_ga)
##                        
##                        # Probabilities
##                        _,_,new_lhood = ppsl_model.kalman_filter(self.observ)
##                        new_givens_prior = ppsl_model.givens_prior()
##                        
##                        ap = (new_lhood+new_givens_prior)-(old_lhood+old_givens_prior)
###                        print(ap)
##                        
##                        # Accept/Reject
##                        if np.log(np.random.random()) < ap:
##                            self.model = ppsl_model
###                            self.givens_acc[gg].append(1)
##                            giv_acc[gg] = 1
##                            old_lhood = new_lhood
##                            pass
##                        else:
###                            self.givens_acc[gg].append(0)
##                            giv_acc[gg] = 0
##                            
##                    self.givens_acc.append(np.mean(giv_acc))
#                    
#                # Sample a new covariance matrix
#                x = self.model.backward_simulation(flt)
#                self.model.D = sysl.sample_degenerate_transition_noise_matrix_conditional(x, self.model.F, self.model.rank, self.model.U, self.Q_prior_dof, self.Q_prior_scale)
#
#    def iterate_transition_noise_matrix_rank(self, Mtimes):
#        """Run RJ-MCMC iterations to change the transition noise matrix rank"""
#        
#        if not(type(self.model)==DegenerateLinearModel):
#            raise TypeError("Q is full rank. Cannot do this sort of move.")
#        
#        if ((self.Q_prior_dof==None) or (self.Q_prior_scale==None)):
#            raise ValueError("A transition covariance matrix prior must be supplied in order to sample")
#            
#        for mm in range(Mtimes):
#            
#            # Likelihood
#            flt,_,old_lhood = self.model.kalman_filter(self.observ)
#            
#            # Copy model
#            ppsl_model = self.model.copy()
#            
#            # Add or remove?
#            if ppsl_model.ds==1:
#                u = -1                          # 1D model. Cannot do anything
#            elif ppsl_model.rank==1:
#                u = 0
#            elif ppsl_model.rank==ppsl_model.ds:
#                u = 1
#            else:
#                u = int(np.round(np.random.random()))
#            
#            if u==0:
#                
#                # Increase rank
#                dcf = ppsl_model.increase_rank(None, None, self.Q_prior_scale)
#                
#            elif u==1:
#                # Decrease rank
#                dcf,val,vec = ppsl_model.reduce_rank(self.Q_prior_scale)
#            
#            # Likelihood
#            _,_,new_lhood = ppsl_model.kalman_filter(self.observ)
#            
#            ap = (new_lhood-old_lhood) + dcf
#            
#            # Accept/Reject
#            if np.log(np.random.random()) < ap:
#                self.model = ppsl_model
#                self.rank_acc.append(1)
##                old_lhood = new_lhood
#            else:
#                self.rank_acc.append(0)
#                
            

            
#### DISPLAY ###
#def draw_chain_histogram(chain, true_model, burn_in):
#    d = min(4,true_model.ds) # Don't plot too many components - too small
#    figs = []
#    
#    if len(chain)>burn_in:
#        # F
#        f,ax = plt.subplots(d,d,squeeze=False,figsize=(9,9))
#        f.subplots_adjust(wspace=0.3, hspace=0.3)
#        figs.append(f)
#        for rr in range(d):
#            for cc in range(d):
#                ax[rr,cc].locator_params(nbins=2)
#                ax[rr,cc].set_ylim((0,len(chain[burn_in:])))
#                ax[rr,cc].hist([mm.F[rr,cc] for mm in chain[burn_in:]], color='0.5')
#                ax[rr,cc].plot([true_model.F[rr,cc]]*2, [0,len(chain[burn_in:])], '-r')
##                ax[rr,cc].set_xlim((-2,2))
#                
#        # Q
#        f,ax = plt.subplots(d,d,squeeze=False,figsize=(9,9))
#        f.subplots_adjust(wspace=0.3, hspace=0.3)
#        figs.append(f)
#        for rr in range(d):
#            for cc in range(d):
#                ax[rr,cc].locator_params(nbins=2)
#                ax[rr,cc].set_ylim((0,len(chain[burn_in:])))
#                ax[rr,cc].hist([mm.Q[rr,cc] for mm in chain[burn_in:]], color='0.5')
#                ax[rr,cc].plot([true_model.Q[rr,cc]]*2, [0,len(chain[burn_in:])], '-r')
##                ax[rr,cc].set_xlim((-2,2))
#        
#        # |Q|
#        f,ax = plt.subplots(1,1)
#        figs.append(f)
#        ax.locator_params(nbins=2)
#        ax.set_ylim((0,len(chain[burn_in:])))
#        ax.hist([la.det(mm.Q) for mm in chain[burn_in:]], color='0.5')
#        ax.plot([la.det(true_model.Q)]*2, [0,len(chain[burn_in:])], '-r')
#        
#        if type(true_model) == SparseLinearModel:
#            
#            # B
#            f,ax = plt.subplots(d,d,squeeze=False,figsize=(9,9))
#            f.subplots_adjust(wspace=0.3, hspace=0.3)
#            figs.append(f)
#            for rr in range(d):
#                for cc in range(d):
#                    ax[rr,cc].locator_params(nbins=2)
#                    ax[rr,cc].set_ylim((0,len(chain[burn_in:])))
#                    ax[rr,cc].hist([mm.B[rr,cc] for mm in chain[burn_in:]],bins=[-0.1,0.1,0.9,1.1], color='0.5')
#                    ax[rr,cc].set_xlim((-0.1,1.1))
#                    ax[rr,cc].plot([true_model.B[rr,cc]]*2, [0,1.2*len(chain[burn_in:])], '-r')
#        
#        if type(true_model) == DegenerateLinearModel:
#        
#            mode_rank = int(stats.mode([mm.rank for mm in chain[burn_in:]])[0][0])
#            
#            # rank
#            f,ax = plt.subplots(1,1)
#            figs.append(f)
#            ax.locator_params(nbins=2)
#            ax.set_ylim((0,len(chain[burn_in:])))
#            rank_chain =np.array([mm.rank for mm in chain[burn_in:]])
#            bins = np.arange(rank_chain.min()-0.5,rank_chain.max()+1.5)            
#            ax.hist(rank_chain, color='0.5', bins=bins)
#            ax.plot([true_model.rank]*2, [0,len(chain[burn_in:])], '-r')
#            
#            # D
#            d = min(3,true_model.rank)
#            f,ax = plt.subplots(d,d,squeeze=False,figsize=(9,9))
#            f.subplots_adjust(wspace=0.3, hspace=0.3)
#            figs.append(f)
#            for rr in range(d):
#                for cc in range(d):
#                    ax[rr,cc].locator_params(nbins=2)
#                    ax[rr,cc].set_ylim((0,len(chain[burn_in:])))
#                    ax[rr,cc].hist([mm.D[rr,cc] for mm in chain[burn_in:] if mm.rank==mode_rank], color='0.5')
#                    ax[rr,cc].plot([true_model.D[rr,cc]]*2, [0,len(chain[burn_in:])], '-r')
#            
#            # givens
#            f,ax = plt.subplots(1,len(true_model.givens),squeeze=False,figsize=(9,5))
#            f.subplots_adjust(wspace=0.3, hspace=0.3)
#            figs.append(f)
#            for cc in range(len(true_model.givens)):
#                ax[0,cc].locator_params(nbins=2)
#                ax[0,cc].set_ylim((0,len(chain[burn_in:])))
#                ax[0,cc].hist([mm.givens[cc] for mm in chain[burn_in:] if mm.rank==mode_rank], color='0.5')
#                ax[0,cc].plot([true_model.givens[cc]]*2, [0,len(chain[burn_in:])], '-r')
##                ax[0,cc].set_xlim((-np.pi,np.pi))
#    
#    return figs
#
#
#def draw_chain(chain, true_model, burn_in, fields):
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
#                        ax[rr,cc].locator_params(nbins=2)
#                        ax[rr,cc].set_xlim((0,len(chain[1:])))
#                        ax[rr,cc].plot([getattr(mm,ff)[rr,cc] for mm in chain[1:]], 'k')
#                        ax[rr,cc].plot([1,len(chain)], [getattr(true_model,ff)[rr,cc]]*2, '-r')
#            elif ff in ['givens']:
#                d = len(getattr(true_model,ff))
#                fig = plt.figure()
#                for dd in range(d):
#                    ax = fig.add_subplot(d,1,dd+1)
#                    ax.plot([getattr(mm,ff)[dd] for mm in chain[1:]])
#                    ax.plot([1,len(chain)], [getattr(true_model,ff)[dd]]*2, 'r')
#                    
#    return fig
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
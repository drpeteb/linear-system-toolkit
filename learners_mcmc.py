# Operational modules
from abc import ABCMeta
import pickle

import numpy as np
import scipy.linalg as la
from scipy import stats
from scipy import special
from statsmodels.tsa import stattools
import matplotlib.pyplot as plt

# Import from other module files
import sampling as smp


def load_learner(filename):
    """
    Loads a pickled learner object.
    """
    fileOb = open(filename, 'rb')
    learner = pickle.load(fileOb)
    fileOb.close()
    return learner


class BaseMCMCLearner():
    __metaclass__ = ABCMeta
    """
    Basic methods for MCMC system learning algorithms.
    Sub-class this and implement methods to perform individual MCMC sampling
    steps. These will depend on the type of linear model and the choice of
    prior distribusion.
    """

    def __init__(self, initial_model_estimate, observ, hyperparams,
                                            algoparams=dict(), verbose=False):
        self.observ = observ
        self.hyperparams = hyperparams

        self.algoparams = algoparams

        self.model = initial_model_estimate
        self.flt,_,self.lhood = self.model.kalman_filter(self.observ)
        self.state = self.model.backward_simulation(self.flt)
        self.filter_current = True

        self.chain_model = []
        self.chain_state = []
        self.chain_lhood = []

        self.chain_accept = dict()
        self.chain_algoparams = dict()

        self.verbose = verbose

    def save(self, filename):
        """
        Pickle and save the object.
        """
        fileOb = open(filename, 'wb')
        pickle.dump(self, fileOb)
        fileOb.close()


    def save_link(self):
        """Save the current state of the model as a link in the chain"""

        # Need to make sure the likelihood value is current
        if not self.filter_current:
            self.flt,_,self.lhood = self.model.kalman_filter(self.observ)
            self.filter_current = True

        self.chain_model.append(self.model.parameters.copy())
        self.chain_state.append(self.state.copy())
        self.chain_lhood.append(self.lhood)


    def adapt_algorithm_parameters(self, batchLength=20, targetRate=0.44,
                                                                minReact=0.1):
        """
        Adapt algorithm parameters based on recent acceptance history.
        """
        for param in list(self.algoparams.keys()):
            length = np.minimum(batchLength,len(self.chain_accept[param]))
            actualRate = np.mean(np.array(self.chain_accept[param][-length:]))
            expnt = np.min(minReact,1.0/np.sqrt(length))
            if actualRate < targetRate:
                expnt *= -1
            self.algoparams[param] *= np.exp(expnt)


    def sample_state_trajectory(self):
        """
        Sample a state trajectory conditional on the current model parameters
        """
        # Need to make sure the likelihood value is current
        if not self.filter_current:
            self.flt,_,self.lhood = self.model.kalman_filter(self.observ)
            self.filter_current = True
        self.state = self.model.backward_simulation(self.flt)

    def estimate_state_trajectory(self, numBurnIn=0):
        """
        Estimate of the state trajectory (mean and standard deviation) using
        all the samples in the chain
        """
        samples = np.array(self.chain_state[numBurnIn:])
        mn = np.mean(samples, axis=0)
        sd = np.std(samples, axis=0)
        return mn, sd


    def _create_2d_plot_axes(self, param, index1=None, index2=None):
        """
        Create a figure, axes, and coordinates for elements of a 2D array
        """

        # Get the parameter shape
        paramShape = param.shape

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

    def _create_1d_plot_axes(self, param, index=None):
        """
        Create a figure, axes, and coordinates for elements of a 1D array
        """

        # Get the parameter shape
        paramShape = param.shape

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


    def plot_chain_accept(self, paramName=None, numBurnIn=0):
        """
        Make Markov chain accpetance plots for algorithm parameters
        """

        if paramName is None:
            paramName = list(self.chain_accept.keys())
        numParams = len(paramName)

        fig, axs = plt.subplots(nrows=1, ncols=numParams, squeeze=False)

        for ii in range(numParams):
            samples = self.chain_accept[paramName[ii]]
            axs[0,ii].plot(np.cumsum(samples), 'k')
            axs[0,ii].set_title(paramName[ii])

        plt.show()


    def plot_chain_adapt(self, paramName=None, numBurnIn=0):
        """
        Make Markov chain adaptive algorithm parameters
        """

        if paramName is None:
            paramName = list(self.chain_algoparams.keys())
        numParams = len(paramName)

        fig, axs = plt.subplots(nrows=1, ncols=numParams, squeeze=False)

        for ii in range(numParams):
            samples = self.chain_algoparams[paramName[ii]]
            axs[0,ii].plot(samples, 'k')
            axs[0,ii].set_title(paramName[ii])

        plt.show()


    def plot_chain_trace(self, paramName, numBurnIn=0, dims=None,
                                               trueModel=None, derived=False):
        """
        Make Markov chain trace plots for a chosen parameter

        dims is a list or tuple of two lists which specificy which rows and
        columns should be plotted. If empty then all are plotted.
        """

        # Get a list of parameters
        if not derived:
            paramList = [md[paramName] for md in self.chain_model]
        else:
            raise NotImplementedError("Doesn't work because of the change"
                                      "in the way the chain is stored.")
#            #TODO Fix this
#            paramList = eval("[md.{}() for md in self.chain_model]"\
#                                                           .format(paramName))

        # Get the true value
        if trueModel is not None:
            if not derived:
                trueValue = trueModel.parameters[paramName]
            else:
                trueValue = eval("trueModel.{}()".format(paramName))
        else:
            trueValue = None

        # Get the parameter shape
        paramShape = paramList[0].shape

        if len(paramShape) == 1:
            fig, axs, coords = self._create_1d_plot_axes(paramList[0], dims)
        elif len(paramShape) == 2:
            if dims is None:
                dims = (None,None)
            fig, axs, coords = self._create_2d_plot_axes(paramList[0],
                                                             dims[0], dims[1])
        else:
            raise ValueError("Cannot draw plots for this parameter")

        for idx in np.ndindex(coords.shape):
            samples = [pp[coords[idx]] for pp in paramList]
            axs[idx].plot(samples, 'k')
            ylims = axs[idx].get_ylim()
            axs[idx].plot([numBurnIn]*2, ylims, 'k:')
            axs[idx].set_ylim(ylims)
            if trueValue is not None:
                axs[idx].plot([0,len(samples)-1],[trueValue[coords[idx]]]*2,
                                                             'r', linewidth=2)

        return fig, axs


    def plot_chain_histogram(self, paramName, numBurnIn=0, dims=None,
                                               trueModel=None, derived=False):
        """
        Make Markov chain histograms for a chosen parameter

        dims is a tuple of two lists specificy which rows and columns should
        be plotted. If empty then all are plotted.
        """

        # Get a list of parameters
        if not derived:
            paramList = [md[paramName] for md in self.chain_model]
        else:
            raise NotImplementedError("Doesn't work because of the change"
                                      "in the way the chain is stored.")
#            #TODO Fix this
#            paramList = eval("[md.{}() for md in self.chain_model]"\
#                                                           .format(paramName))

        # Get the true value
        if trueModel is not None:
            if not derived:
                trueValue = trueModel.parameters[paramName]
            else:
                trueValue = eval("trueModel.{}()".format(paramName))
        else:
            trueValue = None

        # Get the parameter shape
        paramShape = paramList[0].shape

        if len(paramShape) == 1:
            fig, axs, coords = self._create_1d_plot_axes(paramList[0], dims)
        elif len(paramShape) == 2:
            if dims is None:
                dims = (None,None)
            fig, axs, coords = self._create_2d_plot_axes(paramList[0],
                                                             dims[0], dims[1])
        else:
            raise ValueError("Cannot draw plots for this parameter")

        for idx in np.ndindex(coords.shape):
            samples = [pp[coords[idx]] for pp in paramList[numBurnIn:]]
            axs[idx].hist(samples, color='0.8')
            if trueValue is not None:
                ylims = axs[idx].get_ylim()
                axs[idx].plot([trueValue[coords[idx]]]*2, ylims, 'r',
                                                                  linewidth=2)
                axs[idx].set_ylim(ylims)

        return fig, axs


    def plot_chain_acf(self, paramName, numBurnIn=0, dims=None, nlags=30,
                                                               derived=False):
        """
        Make Markov chain autocorrelation function plots for a chosen
        parameter

        dims is a list or tuple of two lists which specificy which rows and
        columns should be plotted. If empty then all are plotted.
        """

        nlags = int(np.minimum( nlags,
                                   np.sqrt(len(self.chain_model)-numBurnIn) ))

        # Get a list of parameters
        if not derived:
            paramList = [md[paramName] for md in self.chain_model]
        else:
            raise NotImplementedError("Doesn't work because of the change"
                                      "in the way the chain is stored.")
#            #TODO Fix this
#            paramList = eval("[md.{}() for md in self.chain_model]"\
#                                                           .format(paramName))
        # Get the parameter shape
        paramShape = paramList[0].shape

        if len(paramShape) == 1:
            fig, axs, coords = self._create_1d_plot_axes(paramList[0], dims)
        elif len(paramShape) == 2:
            if dims is None:
                dims = (None,None)
            fig, axs, coords = self._create_2d_plot_axes(paramList[0],
                                                             dims[0], dims[1])
        else:
            raise ValueError("Cannot draw plots for this parameter")

        for idx in np.ndindex(coords.shape):
            samples = [pp[coords[idx]] for pp in paramList[numBurnIn:]]
            acf = stattools.acf(samples,unbiased=False,nlags=nlags)
            axs[idx].plot(acf, 'k')
            axs[idx].plot([0,nlags], [0,0], 'k:')
            axs[idx].set_xlim([0,nlags])

        return fig, axs


class MCMCLearnerObservationDiagonalCovarianceWithIGPrior():
    __metaclass__ = ABCMeta
    """
    Container for MCMC system learning algorithm.
    Model Type: Basic
    Unknown Parameters: R (observation covariance, assumed diagonal)
    Prior Type: Inverse Gamma
    """

    def sample_observation_diagonal_covariance(self):
        """
        MCMC iteration (Gibbs sampling) for diagonal observation covariance
        matrix with inverse-gamma prior
        """

        # Calculate sufficient statistics using current state trajectory
        suffStats = smp.evaluate_observation_sufficient_statistics(self.state,
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

        # self.flt and self.lhood are no longer up-to-date
        self.filter_current = False


class MCMCLearnerNCVMTransitionCovarianceWithIGPrior():
    __metaclass__ = ABCMeta
    """
    Container for MCMC system learning algorithm.
    Model Type: Basic
    Unknown Parameters: Q (transition covariance, assumed scaled NCVM)
    Prior Type: Inverse Gamma
    """

    def sample_transition_ncvm_covariance(self):
        """
        MCMC iteration (Gibbs sampling) for diagonal observation covariance
        matrix with inverse-gamma prior
        """

        # Calculate sufficient statistics using current state trajectory
        suffStats = smp.evaluate_transition_sufficient_statistics(self.state)

        # Update hyperparameters
        Qs = self.model.parameters['Q']
        Qs /= Qs[0,0]
        a,b = smp.hyperparam_update_ncvm_ig_transition_variance(
                                    suffStats,
                                    self.model.parameters['F'],
                                    Qs,
                                    self.hyperparams['a0'],
                                    self.hyperparams['b0'])  # This uses the same hyperparameters as the observation variance because I'm lazy and want to go to bed.

        # Sample new parameter
        s = stats.invgamma.rvs(a, scale=b)
        self.model.parameters['Q'] = s*Qs

        # self.flt and self.lhood are no longer up-to-date
        self.filter_current = False


class MCMCLearnerTransitionBasicModelWithMNIWPrior():
    __metaclass__ = ABCMeta
    """
    Container for MCMC system learning algorithm.
    Model Type: Basic
    Unknown Parameters: F,Q
    Prior Type: Matrix Normal-Inverse Wishart
    """

    def sample_transition(self):
        """
        MCMC iteration (Gibbs sampling) for transition matrix and covariance
        """

        # Calculate sufficient statistics using current state trajectory
        suffStats = smp.evaluate_transition_sufficient_statistics(self.state)

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

        # self.flt and self.lhood are no longer up-to-date
        self.filter_current = False


class MCMCLearnerTransitionDegenerateModelWithMNIWPrior():
    __metaclass__ = ABCMeta
    """
    Container for MCMC system learning algorithm.
    Model Type: Degenerate
    Unknown Parameters: F,Q,rank(Q)
    Prior Type: Singular Matrix Normal-Inverse Wishart
    """

    def transition_prior(self, model):
        """
        Prior density for transition model parameters
        """
        Psi0 = model.parameters['rank'][0]*self.hyperparams['rPsi0']
        variancePrior = smp.singular_inverse_wishart_density(
                                            model.parameters['val'],
                                            model.parameters['vec'],
                                            Psi0)

        orthVec = model.complete_basis()
        relaxEval = self.hyperparams['alpha']
#        relaxEval = np.min(model.parameters['val'])
#        relaxEval = np.max(model.parameters['val'])
        rowVariance = model.transition_covariance() \
                      + relaxEval*np.dot(orthVec,orthVec.T)
        matrixPrior = smp.matrix_normal_density(model.parameters['F'],
                                                self.hyperparams['M0'],
                                                rowVariance,
                                                self.hyperparams['V0'])

        return variancePrior + matrixPrior


    def sample_transition_covariance(self, moveType):
        """
        MCMC iteration (Metropolis-Hastings) for transition covariance.
        """

        # Make sure there's a dictionary to store stats for this move type
        if moveType not in self.chain_accept:
            self.chain_accept[moveType] = []
        if moveType not in self.chain_algoparams:
            self.chain_algoparams[moveType] = []

        # Need to make sure the likelihood value is current
        if not self.filter_current:
            self.flt,_,self.lhood = self.model.kalman_filter(self.observ)
            self.filter_current = True

        # Copy model
        ppsl_model = self.model.copy()

        if self.verbose:
            print("Metropolis-Hastings move for transition covariance. "
                  "Type: {}".format(moveType))

        # Propose change
        if moveType == 'rotate':

            # Make the change
            rotation = smp.sample_cayley(self.model.ds,
                                                    self.algoparams[moveType])
            ppsl_model.rotate_transition_covariance(rotation)

            # Random walk, so forward and backward probabilities are same
            fwd_prob = 0
            bwd_prob = 0

            self.chain_algoparams[moveType].append(self.algoparams[moveType])

        elif moveType == 'rank':
            # Assumes uniform prior on each possible value of the rank

            if ppsl_model.ds == 1:
                switch = -1                               # Do nothing (1D)
            elif ppsl_model.parameters['rank'][0] == 1:
                switch = 1                                # Must increase rank
            elif ppsl_model.parameters['rank'][0] == ppsl_model.ds:
                switch = 0                                # Must decrease rank
            else:
                switch = np.random.random_integers(0,1)   # choose at random

            self.chain_algoparams[moveType].append(None)

            if switch == -1:
                # No options to change rank

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
                valPpslProb = -np.log(minValue)

                # Nullspace
                nullSpace = ppsl_model.complete_basis()
                nullDims = nullSpace.shape[1]

                # Calculate the Jacobian
                constJac = 0.5*nullDims*np.log(np.pi) \
                         - np.sum(np.log(ppsl_model.parameters['val'])) \
                         - special.gammaln(nullDims/2)
                varJac = (nullDims-1)*np.log(oldValue) \
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
                valPpslProb = -np.log(minValue)

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
                varJac = (nullDims-1)*np.log(newValue) \
                       + np.sum(np.log(ppsl_model.parameters['val']-newValue))
                                # Add them to the model
                ppsl_model.add_eigen_value_vector(newValue, newVector)

                # Fudge the proposals and jacobian into the proposal terms
                fwd_prob = valPpslProb
                bwd_prob = constJac + varJac

        else:
            raise ValueError("Invalid move type")

        # Kalman filter
        ppsl_flt,_,ppsl_lhood = ppsl_model.kalman_filter(self.observ)

        # Prior terms
        prior = self.transition_prior(self.model)
        ppsl_prior = self.transition_prior(ppsl_model)

#        print(ppsl_lhood-self.lhood)
#        print(ppsl_prior-prior)
#        print(bwd_prob-fwd_prob)

        # Decide
        acceptRatio =   (ppsl_lhood-self.lhood) \
                      + (ppsl_prior-prior) \
                      + (bwd_prob-fwd_prob)
        if self.verbose:
            print("   Acceptance ratio: {}".format(acceptRatio))
        if np.log(np.random.random()) < acceptRatio:
            self.model = ppsl_model
            self.flt = ppsl_flt
            self.lhood = ppsl_lhood
            self.chain_accept[moveType].append(True)
            if self.verbose:
                print("   accepted")
        else:
            self.chain_accept[moveType].append(False)
            if self.verbose:
                print("   rejected")


    def sample_transition_matrix(self):
        """
        MCMC iteration (Metropolis-Hastings) for transition matrix
        """

        # Make sure there's a dictionary to store stats for this move type
        moveType = 'perturb'
        if moveType not in self.chain_accept:
            self.chain_accept[moveType] = []
        if moveType not in self.chain_algoparams:
            self.chain_algoparams[moveType] = []

        # Need to make sure the likelihood value is current
        if not self.filter_current:
            self.flt,_,self.lhood = self.model.kalman_filter(self.observ)
            self.filter_current = True

        # Copy model
        ppsl_model = self.model.copy()

        if self.verbose:
            print("Metropolis-Hastings move for transition matrix.")

        # Propose a new transition matrix
        ds = ppsl_model.ds
        I = np.identity(ds)
        sf = smp.sample_matrix_normal(np.zeros((ds,ds)),
                                               self.algoparams[moveType]*I, I)
        ppsl_model.parameters['F'] *= np.exp(sf)
#        ppsl_model.parameters['F'] = smp.sample_matrix_normal(
#                   self.model.parameters['F'], self.algoparams[moveType]*I, I)
        self.chain_algoparams[moveType].append(self.algoparams[moveType])

        # Random walk, so forward and backward probabilities are same
        fwd_prob = 0
        bwd_prob = 0

#        # Propose a new transition matrix
#        suffStats = smp.evaluate_transition_sufficient_statistics(self.state)
#        padded_Q = ppsl_model.transition_covariance() + \
#                          self.algoparams[moveType]*np.identity(ppsl_model.ds)
#        M,V = smp.hyperparam_update_basic_mn_transition_matrix(
#                                                    suffStats,
#                                                    self.hyperparams['M0'],
#                                                    self.hyperparams['V0'],)
#        ppsl_F = smp.sample_matrix_normal(M,padded_Q,V)
#        fwd_prob = smp.matrix_normal_density(ppsl_F,M,padded_Q,V)
#        ppsl_model.parameters['F'] = ppsl_F
#        self.chain_algoparams[moveType].append(self.algoparams[moveType])
#
#        # Sample a new trajectory
#        ppsl_state = ppsl_model.sample_posterior(self.observ)
#        ppsl_suffStats = smp.evaluate_transition_sufficient_statistics(
#                                                                   ppsl_state)
#
#        # Reverse move probaility
#        M,V = smp.hyperparam_update_basic_mn_transition_matrix(
#                                                    ppsl_suffStats,
#                                                    self.hyperparams['M0'],
#                                                    self.hyperparams['V0'],)
#        bwd_prob = smp.matrix_normal_density(self.model.parameters['F'],
#                                                                 M,padded_Q,V)

        # Kalman filter
        ppsl_flt,_,ppsl_lhood = ppsl_model.kalman_filter(self.observ)

        # Prior terms
        prior = self.transition_prior(self.model)
        ppsl_prior = self.transition_prior(ppsl_model)

#        print(ppsl_lhood-self.lhood)
#        print(ppsl_prior-prior)

        # Decide
        acceptRatio =   (ppsl_lhood-self.lhood) \
                      + (ppsl_prior-prior) \
                      + (bwd_prob-fwd_prob)
        if self.verbose:
            print("   Acceptance ratio: {}".format(acceptRatio))
        if np.log(np.random.random()) < acceptRatio:
            self.model = ppsl_model
            self.flt = ppsl_flt
            self.lhood = ppsl_lhood
#            self.state = ppsl_state
            self.chain_accept[moveType].append(True)
            if self.verbose:
                print("   accepted")
        else:
            self.chain_accept[moveType].append(False)
            if self.verbose:
                print("   rejected")


    def sample_transition_within_subspace(self):
        """
        MCMC iteration (Gibbs sampling) for transition matrix and covariance
        within the constrained subspace
        """

        # Calculate sufficient statistics
        suffStats = smp.evaluate_transition_sufficient_statistics(self.state)

        # Convert to Givens factorisation form
        U,D = self.model.convert_to_givens_form()

        # Sample a new projected transition matrix and transition covariance
        rank = self.model.parameters['rank'][0]
        nu0 = rank
        Psi0 = rank*self.hyperparams['rPsi0']
        nu,Psi,M,V = smp.hyperparam_update_degenerate_mniw_transition(
                                                    suffStats, U,
                                                    nu0,
                                                    Psi0,
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

        # self.flt and self.lhood are no longer up-to-date
        self.filter_current = False

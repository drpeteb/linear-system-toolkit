# Operational modules
import pickle

import numpy as np
import scipy.linalg as la
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt

# Import from other module files
import sampling as smp
from linear_models import DegenerateLinearModel


def complete_basis(vec):
    """
    Returns a matrix of vectors orthogonal to the eigenvectors for the
    transition covariance
    """
    d,r = vec.shape
    Q,_ = la.qr(vec)
    return Q[:,r:]


def effective_sample_size(weight):
    """
    Calculate effective sample size for a set of unnormalised importance
    log-weights.
    """
    w = weight.copy()
    w -= np.max(w)
    w = np.exp(w)
    w /= np.sum(w)
    return 1.0/(np.sum(w**2))


def normalising_constant_estimate(weight):
    """
    Estimate normalising constant from a set of unnormalised importance
    log-weights
    """
    w = weight.copy()
    wmax = np.max(w)
    w -= wmax
    w = np.exp(w)
    return np.log(np.sum(w)) + wmax


def transition_prior(rank, val, vec, F, hyperparams):
    """
    Prior density for transition model parameters
    """
    Psi0 = rank*hyperparams['rPsi0']
    variancePrior = smp.singular_inverse_wishart_density(val, vec, Psi0)

    orthVec = complete_basis(vec)
#    relaxEval = self.hyperparams['alpha']
#    relaxEval = np.min(model.parameters['val'])
    relaxEval = np.max(val)
    rowVariance = np.dot(vec, np.dot(np.diag(val), vec.T)) \
                  + relaxEval*np.dot(orthVec,orthVec.T)
    matrixPrior = smp.matrix_normal_density(F,
                                            hyperparams['M0'],
                                            rowVariance,
                                            hyperparams['V0'])

    return variancePrior + matrixPrior


def extended_density(extraVal, extraVec, val):
    """
    Artificial conditional 'extension' density for extra eigenvalue/vector.
    This is currently uniform over the interval [0,l] for the eigenvalue
    (where l is the next largest eigenvalue) and uniform (i.e. Haar) for the
    eigenvector
    """
    r = val.shape[0]
    d = extraVec.shape[0]
    valProb = -np.log(np.min(val))
    vecProb = special.gammaln(0.5*(d-r)) - 0.5*(d-r)*np.log(np.pi)
    return valProb + vecProb


def sample_transition_within_subspace(model, state, hyperparams):
    """
    MCMC iteration (Gibbs sampling) for transition matrix and covariance
    within the constrained subspace
    """

    # Calculate sufficient statistics
    suffStats = smp.evaluate_transition_sufficient_statistics(state)

    # Convert to Givens factorisation form
    U,D = model.convert_to_givens_form()

    # Sample a new projected transition matrix and transition covariance
    rank = model.parameters['rank'][0]
    nu0 = rank
    Psi0 = rank*hyperparams['rPsi0']
    nu,Psi,M,V = smp.hyperparam_update_degenerate_mniw_transition(
                                                suffStats, U,
                                                nu0,
                                                Psi0,
                                                hyperparams['M0'],
                                                hyperparams['V0'])
    D = la.inv(smp.sample_wishart(nu, la.inv(Psi)))
    FU = smp.sample_matrix_normal(M, D, V)

    # Project out
    Fold = model.parameters['F']
    F = smp.project_degenerate_transition_matrix(Fold, FU, U)
    model.parameters['F'] = F

    # Convert back to eigen-decomposition form
    model.update_from_givens_form(U, D)

    return model


def sample_observation_diagonal_covariance(model, state, observ, hyperparams):
    """
    MCMC iteration (Gibbs sampling) for diagonal observation covariance
    matrix with inverse-gamma prior
    """

    # Calculate sufficient statistics using current state trajectory
    suffStats = smp.evaluate_observation_sufficient_statistics(state, observ)

    # Update hyperparameters
    a,b = smp.hyperparam_update_basic_ig_observation_variance(
                                suffStats,
                                model.parameters['H'],
                                hyperparams['a0'],
                                hyperparams['b0'])

    # Sample new parameter
    r = stats.invgamma.rvs(a, scale=b)
    model.parameters['R'] = r*np.identity(model.do)

    return model


class DegenerateModelSMCApproximation():
    """
    Class to hold and provide access to the elements of an SMC approximation
    of a degenerate linear model.
    """

    def __init__(self, N, d, r):
        self.rank = r
        self.ds = d
        self.F = np.zeros((N,d,d))
        self.val = np.zeros((N,r))
        self.vec = np.zeros((N,d,r))
        self.Rs = np.zeros((N))
        self.weight = np.zeros((N))
        self.prior = np.zeros((N))
        self.lhood = np.zeros((N))



class DegenerateSMCLearner():
    """
    SMC learning class for degenerate linear state space models. This is
    specifically written for the mocap problem in the paper.
    """
    def save(self, filename):
        """
        Pickle and save the object.
        """
        fileOb = open(filename, 'wb')
        pickle.dump(self, fileOb)
        fileOb.close()

    def __init__(self, chain_model, chain_lhood, chain_state, observ,
                 hyperparams, initial_state_prior, num_rejuv, verbose=False):
        """
        Take the output of an MCMC algorithm assuming a full-rank model and
        convert it into an SMC approximation.
        """

        self.approx = dict()
        self.state = dict()
        N = len(chain_model)
        K,do = observ.shape
        d = chain_model[0]['F'].shape[0]
        r = chain_model[0]['val'].shape[0]
        self.approx[d] = DegenerateModelSMCApproximation(N, d, r)
        self.state[d] = np.zeros((N, 1, K, d)) # Note that the (r-1)th entry of state stores samples corresponding to the (r)th rank
        for nn in range(N):
            self.state[d][nn,0,:,:] = chain_state[nn].copy()
            self.approx[d].F[nn,:,:] = chain_model[nn]['F'].copy()
            self.approx[d].val[nn,:] = chain_model[nn]['val'].copy()
            self.approx[d].vec[nn,:,:] = chain_model[nn]['vec'].copy()
            self.approx[d].Rs[nn] = chain_model[nn]['R'][0,0].copy()
            self.approx[d].lhood[nn] = chain_lhood[nn].copy()
            self.approx[d].prior[nn] = transition_prior(r,
                                                        chain_model[nn]['val'],
                                                        chain_model[nn]['vec'],
                                                        chain_model[nn]['F'],
                                                        hyperparams)


        self.filters = None

        self.observ = observ
        self.hyperparams = hyperparams
        self.initial_state_prior = initial_state_prior
        self.verbose = verbose
        self.num_samples = N
        self.num_rejuv = num_rejuv
        self.ds = d
        self.do = do
        self.K = K
        self.H = chain_model[0]['H']



    def smc_reduce_rank(self, rank):
        """
        The main step of the algorithm. Use the previous approximation to
        'propose' parameters for a reduced rank mode, and weight them
        correctly.
        """

        # Create a new SMC approximation
        if rank in self.approx.keys():
            raise ValueError("Already done that one")
        if rank+1 not in self.approx.keys():
            raise ValueError("Need to do rank {} first.".format(rank+1))
        self.approx[rank] = DegenerateModelSMCApproximation(self.num_samples,
                                                            self.ds, rank)

        # Create space to store the state trajectories
        self.state[rank] = np.zeros((self.num_samples,
                                     self.num_rejuv,
                                     self.K,
                                     self.ds))

        # Create a store for the filter results for RM in the next iteration
        filters = []

        # Resampling
        w = self.approx[rank+1].weight.copy()
        w -= np.max(w)
        w = np.exp(w)
        w /= np.sum(w)
        ancestors = np.random.choice(self.num_samples,
                                     size=self.num_samples,
                                     replace=True,
                                     p=w)
        self.approx[rank].ancestor = ancestors

        # Loop through samples
        for nn in range(self.num_samples):

            if self.verbose:
                print("Sample number {}.".format(nn+1))

            # Create model object
            ai = ancestors[nn]
            parameters = {
                      'F': self.approx[rank+1].F[ai,:,:].copy(),
                      'rank': [rank+1],
                      'val': self.approx[rank+1].val[ai,:].copy(),
                      'vec': self.approx[rank+1].vec[ai,:,:].copy(),
                      'H': self.H,
                      'R': self.approx[rank+1].Rs[ai].copy()*np.identity(self.do)
                      }

            model = DegenerateLinearModel(self.ds,
                                          self.do,
                                          self.initial_state_prior,
                                          parameters)

            # Resample-move with Gibbs sampling to improve diversity
            if (self.filters is not None) and (self.num_rejuv > 0):
                flt = self.filters[ai]
                for ii in range(self.num_rejuv):
                    state = model.backward_simulation(flt)
                    model = sample_transition_within_subspace(model, state,
                                                              self.hyperparams)
                    model = sample_observation_diagonal_covariance(
                                                            model,
                                                            state,
                                                            self.observ,
                                                            self.hyperparams)
                    flt,_,old_lhood = model.kalman_filter(self.observ)
                    self.state[rank][nn,ii,:,:] = state
                old_prior = transition_prior(rank,
                                             model.parameters['val'],
                                             model.parameters['vec'],
                                             model.parameters['F'],
                                             self.hyperparams)
            else:
                old_prior = self.approx[rank+1].prior[ai]
                old_lhood = self.approx[rank+1].lhood[ai]


            # Remove smallest eigenvalue/vector pair
            remVal, remVec = model.remove_min_eigen_value_vector()

            # Probabilities for new model
            prior = transition_prior(rank,
                                     model.parameters['val'],
                                     model.parameters['vec'],
                                     model.parameters['F'],
                                     self.hyperparams)
            flt,_,lhood = model.kalman_filter(self.observ)
            exten = extended_density(remVal, remVec, model.parameters['val'])

            # Jacobian of transformation
            jac = - np.log(2) \
                  - np.sum(np.log(model.parameters['val'])) \
                  + (self.ds - rank - 1)*np.log(remVal) \
                  + np.sum(np.log(model.parameters['val']-remVal))

            # Calculate weight
            weight = + prior \
                     + lhood \
                     - old_prior \
                     - old_lhood \
                     - jac \
                     + exten

#            print(lhood-old_lhood)
#            print(prior-old_prior)
#            print(exten)
#            print(jac)

            if self.verbose:
                print("Particle log-weight: {}".format(weight))

            # Store everything
            filters.append(flt)
            self.approx[rank].prior[nn] = prior
            self.approx[rank].lhood[nn] = lhood
            self.approx[rank].weight[nn] = weight
            self.approx[rank].F[nn,:,:] = model.parameters['F'].copy()
            self.approx[rank].val[nn] = model.parameters['val'].copy()
            self.approx[rank].vec[nn] = model.parameters['vec'].copy()
            self.approx[rank].Rs[nn] = model.parameters['R'][0][0]

        # End of particle loop

        # Save the filter results for later
        self.filters = filters

        if self.verbose:
            print("For rank {}, effective sample size: {}".format(rank,
                  effective_sample_size(self.approx[rank].weight)))







    def estimate_state_trajectory(self, rank):
        """
        Estimate of the state trajectory (mean and standard deviation) using
        the samples from a particular rank of covariance matrix
        """
        shape = (self.num_rejuv*self.num_samples, self.K, self.ds)
        samples = np.reshape(self.state[rank-1], shape,order='F')
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



    def plot_chain_trace(self, paramName, numBurnIn=0, dims=None,
                                               trueModel=None, derived=False):
        #TODO Rewrite for SMC
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


    def plot_chain_histogram(self, paramName, numBurnIn=0, dims=None,
                                               trueModel=None, derived=False):
        #TODO Rewrite for SMC
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








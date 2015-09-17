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
from learners_mcmc import BaseMCMCLearner
import sampling as smp


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


def sample_transition_full_rank(model, state, hyperparams, pseudo_dof=None, pseudo_sd=None):
    """
    MCMC iteration (Gibbs sampling) for transition matrix and covariance
    with full rank model
    """

    Q = model.transition_covariance()

    # Sampke a pseudo-observation to constrain the size of move
    if pseudo_dof is not None:
        extra_nu = pseudo_dof
        extra_Psi = smp.sample_wishart(pseudo_dof, Q)
    else:
        extra_nu = 0
        extra_Psi = 0

    # Calculate sufficient statistics
    suffStats = smp.evaluate_transition_sufficient_statistics(state)

    # Sample a new projected transition matrix and transition covariance
    nu0 = model.ds + extra_nu
    Psi0 = model.ds*hyperparams['rPsi0'] + extra_Psi
    nu,Psi,M,V = smp.hyperparam_update_basic_mniw_transition(
                                                suffStats,
                                                nu0,
                                                Psi0,
                                                hyperparams['M0'],
                                                hyperparams['V0'])
    Q = la.inv(smp.sample_wishart(nu, la.inv(Psi)))
    F = smp.sample_matrix_normal(M, Q, V)

    model.parameters['F'] = F
    model.parameters['Q']= Q

    return model


def sample_transition_covariance_full_rank(model, state, hyperparams, pseudo_dof=None):
    """
    MCMC iteration (Gibbs sampling) for transition covariance
    with full rank model
    """

    Q = model.transition_covariance()

    # Sampke a pseudo-observation to constrain the size of move
    if pseudo_dof is not None:
        extra_nu = pseudo_dof
        extra_Psi = smp.sample_wishart(pseudo_dof, Q)
    else:
        extra_nu = 0
        extra_Psi = 0

    # Calculate sufficient statistics
    suffStats = smp.evaluate_transition_sufficient_statistics(state)

    # Sample a new projected transition matrix and transition covariance
    nu0 = model.ds + extra_nu
    Psi0 = model.ds*hyperparams['rPsi0'] + extra_Psi
    nu,Psi,M,V = smp.hyperparam_update_basic_iw_transition_covariance(
                                                suffStats,
                                                model.parameters['F'],
                                                nu0,
                                                Psi0)
    Q = la.inv(smp.sample_wishart(nu, la.inv(Psi)))
    model.parameters['Q']= Q

    return model


def sample_transition_within_subspace(model, state, hyperparams, pseudo_dof=None, pseudo_sd=None):
    """
    MCMC iteration (Gibbs sampling) for transition matrix and covariance
    within the constrained subspace
    """

    # Calculate sufficient statistics
    suffStats = smp.evaluate_transition_sufficient_statistics(state)

    # Convert to Givens factorisation form
    U,D = model.convert_to_givens_form()

    # Sampke a pseudo-observation to constrain the size of move
    if pseudo_dof is not None:
        extra_nu = pseudo_dof
        extra_Psi = smp.sample_wishart(pseudo_dof, D)
    else:
        extra_nu = 0
        extra_Psi = 0

    # Sample a new projected transition matrix and transition covariance
    rank = model.parameters['rank'][0]
    nu0 = rank + extra_nu
    Psi0 = rank*hyperparams['rPsi0'] + np.dot(U, np.dot(extra_Psi, U.T))
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


def sample_transition_covariance_within_subspace(model, state, hyperparams, pseudo_dof=None):
    """
    MCMC iteration (Gibbs sampling) for transition matrix and covariance
    within the constrained subspace
    """

    # Calculate sufficient statistics
    suffStats = smp.evaluate_transition_sufficient_statistics(state)

    # Convert to Givens factorisation form
    U,D = model.convert_to_givens_form()

    # Sampke a pseudo-observation to constrain the size of move
    if pseudo_dof is not None:
        extra_nu = pseudo_dof
        extra_Psi = smp.sample_wishart(pseudo_dof, D)
    else:
        extra_nu = 0
        extra_Psi = 0

    # Sample a new projected transition matrix and transition covariance
    rank = model.parameters['rank'][0]
    nu0 = rank + extra_nu
    Psi0 = rank*hyperparams['rPsi0'] + np.dot(U, np.dot(extra_Psi, U.T))
    nu,Psi = smp.hyperparam_update_degenerate_iw_transition_covariance(
                                                suffStats, U,
                                                model.parameters['F'],
                                                nu0,
                                                Psi0)
    D = la.inv(smp.sample_wishart(nu, la.inv(Psi)))

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


class CleverDegenerateMCMCLearner(BaseMCMCLearner):


    def __init__(self, initial_model_estimate, observ, hyperparams,
                                            algoparams=dict(), verbose=False):
        self.observ = observ
        self.hyperparams = hyperparams

        self.algoparams = algoparams

        self.model = initial_model_estimate.copy()
        self.base_model = initial_model_estimate.copy()
        assert(self.model.parameters['rank'] == self.model.ds)

        self.weights = np.zeros(self.model.ds)
        self.base_flt,_,_ = self.base_model.kalman_filter(self.observ)
        self.flt = self.base_flt
        self.state = None

        self.chain_model = []
        self.chain_state = []

        self.chain_accept = dict()
        self.chain_algoparams = dict()

        self.verbose = verbose

    def save_link(self):
        """Save the current state of the model as a link in the chain"""

        self.chain_model.append(self.model.parameters.copy())
        self.chain_state.append(self.state.copy())

    def sample_transition_warm_up(self):
        state = self.base_model.backward_simulation(self.base_flt)
        self.base_model = sample_transition_within_subspace(self.base_model,
                                                            state,
                                                            self.hyperparams)
        self.base_flt,_,_ = self.base_model.kalman_filter(self.observ)
        self.model = self.base_model.copy()
        self.flt = self.base_flt


#    def sample_transition_constrained(self):
#        state = self.model.backward_simulation(self.flt)
#        self.model = sample_transition_within_subspace(self.model,
#                                                       state,
#                                                       self.hyperparams)
#        self.flt,_,_ = self.model.kalman_filter(self.observ)


    def sample_transition(self, rank):
        """
        Sample a new transition model
        """
        if self.verbose:
            print("Trying rank {}.".format(rank))

        moveType = "clever"
        if moveType not in self.chain_accept:
            self.chain_accept[moveType] = []

        # Copy the base model and set up arrays
        ppsl_base_model = self.base_model.copy()
        weights = np.zeros(self.model.ds)


#        pob = 0

        # Sample a change to the base model
        base_state = ppsl_base_model.backward_simulation(self.base_flt)
        ppsl_base_model = sample_transition_full_rank(
                                ppsl_base_model,
                                base_state,
                                self.hyperparams,
                                self.algoparams['pseudo_dof'])
        ppsl_base_model = sample_observation_diagonal_covariance(
                                                            ppsl_base_model,
                                                            base_state,
                                                            self.observ,
                                                            self.hyperparams)
        ppsl_base_flt,_,old_lhood = ppsl_base_model.kalman_filter(self.observ)
        old_prior = transition_prior(ppsl_base_model.ds,
                                     ppsl_base_model.parameters['val'],
                                     ppsl_base_model.parameters['vec'],
                                     ppsl_base_model.parameters['F'],
                                     self.hyperparams)

        # Create the proposed model
        ppsl_model = ppsl_base_model.copy()

        # Loop down through the possible ranks
        for rr in reversed(range(rank,self.model.ds)):

            # Reduce rank
            remVal, remVec = ppsl_model.remove_min_eigen_value_vector()

            # Probabilities for new model
            prior = transition_prior(rr,
                                     ppsl_model.parameters['val'],
                                     ppsl_model.parameters['vec'],
                                     ppsl_model.parameters['F'],
                                     self.hyperparams)
            flt,_,lhood = ppsl_model.kalman_filter(self.observ)
            exten = extended_density(remVal, remVec,
                                     ppsl_model.parameters['val'])

            # Jacobian of transformation
            jac = - np.log(2) \
                  - np.sum(np.log(ppsl_model.parameters['val'])) \
                  + (self.model.ds - rr - 1)*np.log(remVal) \
                  + np.sum(np.log(ppsl_model.parameters['val']-remVal))

            # Calculate weight
            weights[rr] = + prior \
                          + lhood \
                          - old_prior \
                          - old_lhood \
                          - jac \
                          + exten

#            print(lhood-old_lhood)
#            print(prior-old_prior)
#            print(exten-jac)

            # Keep things for next step
            old_prior = prior
            old_lhood = lhood

            # Gibbs sampling kernel
            state = ppsl_model.backward_simulation(flt)
            ppsl_model = sample_transition_within_subspace(
                                                ppsl_model,
                                                state,
                                                self.hyperparams,
                                                self.algoparams['pseudo_dof'])
#            ppsl_model = sample_transition_covariance_within_subspace(
#                                                ppsl_model,
#                                                state,
#                                                self.hyperparams,
#                                                self.algoparams['pseudo_dof'])
            ppsl_model = sample_observation_diagonal_covariance(
                                                            ppsl_model,
                                                            state,
                                                            self.observ,
                                                            self.hyperparams)

        if rank == self.model.ds:
            flt,_,_ = ppsl_model.kalman_filter(self.observ)

        # Extra Gibbs the last time round
        num_rejuv = 5
        for ii in range(num_rejuv):
            state = ppsl_model.backward_simulation(flt)
            ppsl_model = sample_transition_within_subspace(
                                                ppsl_model,
                                                state,
                                                self.hyperparams,
                                                self.algoparams['pseudo_dof'])
            ppsl_model = sample_observation_diagonal_covariance(
                                                            ppsl_model,
                                                            state,
                                                            self.observ,
                                                            self.hyperparams)
            flt,_,_ = ppsl_model.kalman_filter(self.observ)


        # Calculate acceptance probability
        acceptRatio = np.sum(weights) - np.sum(self.weights)
        print(weights-self.weights)

        if self.verbose:
            print("   Acceptance ratio: {}".format(acceptRatio))
        if np.log(np.random.random()) < acceptRatio:
            self.model = ppsl_model
            self.base_model = ppsl_base_model
            self.base_flt = ppsl_base_flt
            self.flt = flt
            self.weights = weights
            self.state = state
            self.chain_accept[moveType].append(True)
            if self.verbose:
                print("   accepted")
        else:
            self.chain_accept[moveType].append(False)
            if self.verbose:
                print("   rejected")

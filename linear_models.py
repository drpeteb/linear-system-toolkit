# Operational modules
from abc import ABCMeta, abstractmethod
from copy import deepcopy

# Numerical modules
import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn

# Import from other module files
import kalman as kal
import givens as giv
from kalman import GaussianDensityTimeSeries


class AbstractLinearModel:
    __metaclass__ = ABCMeta
    """
    Abstract Linear-Gaussian Model Class.
    This implements various standard procedures (such as filtering and
    sampling) which are common to any parameterisation. Derived classes should
    implement the abstract methods which supply the system matrices. Parameters
    are held in the the parameters dictionary and should each be a numpy array
    (even if they are scalars).
    """
    
    def __init__(self,
                 state_dimension,
                 observation_dimension,
                 initial_state_prior,
                 parameters):
        """Initialise with model parameters"""
        
        # Consistency checks
        ds = state_dimension
        if (state_dimension<1) or (observation_dimension<1):
            raise ValueError("Invalid state or observation dimensions")
        if (initial_state_prior.mn.shape!=(ds,)) \
                                   or (initial_state_prior.vr.shape!=(ds,ds)):
            raise ValueError("Invalid initial state prior density")
        
        # Store them
        self.ds = state_dimension
        self.do = observation_dimension
        self.initial_state_prior = initial_state_prior
        self.parameters = parameters
    
    
    def copy(self):
        return self.__class__(self.ds, self.do, self.initial_state_prior,
                                                    deepcopy(self.parameters))
    
    
    @abstractmethod
    def transition_matrix(self):
        pass
    
    @abstractmethod
    def transition_covariance(self):
        pass
    
    @abstractmethod
    def observation_matrix(self):
        pass
    
    @abstractmethod
    def observation_covariance(self):
        pass
    
    
    def simulate_data(self, num_time_instants):
        """Sample data a priori according to the model"""
        state = self.sample_state(num_time_instants)
        observ = self.sample_observ(state)
        return state, observ


    def sample_state(self, num_time_instants):
        """Sample a sequence of states from the prior"""
        
        # Get system matices
        F = self.transition_matrix()
        Q = self.transition_covariance()
        
        # Initialise state sequence
        state = np.zeros((num_time_instants,self.ds))
        
        # Sample first value from prior
        state[0]  = mvn.rvs( mean=self.initial_state_prior.mn,
                             cov =self.initial_state_prior.vr )
        
        # Loop through time, sampling each state
        for kk in range(1,num_time_instants):
            state[kk]  = mvn.rvs( mean=np.dot(F,state[kk-1]), cov=Q )
        
        return state


    def sample_observ(self, state):
        """Sample a sequence of observations from the prior"""
        
        # Get system matrices
        H = self.observation_matrix()
        R = self.observation_covariance()
        
        # Initialise observation sequence
        num_time_instants = len(state)
        observ = np.zeros((num_time_instants,self.do))
        
        # Loop through time, sampling each observation
        for kk in range(num_time_instants):
            observ[kk] = mvn.rvs( mean=np.dot(H,state[kk]), cov=R )
        
        return observ
    
    
    def kalman_filter(self, observ):
        """Kalman filter using the model on a set of observations"""
        
        # Get system matrices
        F = self.transition_matrix()
        Q = self.transition_covariance()
        H = self.observation_matrix()
        R = self.observation_covariance()
        
        # Initialise arrays of Gaussian densities and (log-)likelihood
        num_time_instants = len(observ)
        flt = GaussianDensityTimeSeries(num_time_instants, self.ds)
        prd = GaussianDensityTimeSeries(num_time_instants, self.ds)
        lhood = 0
        
        # Loop through time instants
        for kk in range(num_time_instants):
            
            # Prediction
            if kk > 0:
                prd_kk = kal.predict(flt.get_instant(kk-1), F, Q)
            else:
                prd_kk = self.initial_state_prior
            prd.set_instant(kk, prd_kk)
            
            # Correction - skip if there are NaNs, indicating missing data
            y = observ[kk]
            if not np.any(np.isnan(y)):
                flt_kk,innov = kal.correct(prd.get_instant(kk), y, H, R)
                flt.set_instant(kk, flt_kk)
                lhood = lhood + mvn.logpdf(observ[kk], innov.mn, innov.vr)
            
        return flt, prd, lhood

    def rts_smoother(self, flt, prd):
        """Rauch-Tung-Striebel smooth using the model"""
        
        # Get system matrices
        F = self.transition_matrix()
        
        # Initialise arrays of Gaussian densities and (log-)likelihood
        num_time_instants = flt.num_time_instants
        smt = GaussianDensityTimeSeries(num_time_instants, self.ds)
        
        # Loop through time instants
        for kk in reversed(range(num_time_instants)):
            
            # RTS update
            if kk<num_time_instants-1:
                smt_kk = kal.update(flt.get_instant(kk), smt.get_instant(kk+1), 
                                                      prd.get_instant(kk+1), F)
            else:
                smt_kk = flt.get_instant(kk)
            smt.set_instant(kk, smt_kk)
            
        return smt
        
    def backward_simulation(self, flt):
        """Use backward simulation to sample from the state joint posterior"""
        
        # Get system matrices
        F = self.transition_matrix()
        Q = self.transition_covariance() + 1E-10*np.identity(self.ds)
        
        # Initialise sampled sequence
        num_time_instants = flt.num_time_instants
        x = np.zeros((num_time_instants, self.ds))
        
        # Loop through time instatnts, sampling each state
        for kk in reversed(range(num_time_instants)):
            if kk < num_time_instants-1:
                samp_dens,_ = kal.correct(flt.get_instant(kk), x[kk+1], F, Q)
            else:
                samp_dens = flt.get_instant(kk)
            x[kk] = mvn.rvs(mean=samp_dens.mn, cov=samp_dens.vr)
            
        return x

    def sample_posterior(self, observ):
        """Sample a state trajectory from the joint smoothing distribution"""
        flt,_,_ = self.kalman_filter(observ)
        x = self.backward_simulation(flt)
        return x



class BasicLinearModel(AbstractLinearModel):
    """
    Basic linear model where the four system matrices are specified directly.
    """
    
    def transition_matrix(self):
        return self.parameters['F']

    def transition_covariance(self):
        return self.parameters['Q']
        
    def observation_matrix(self):
        return self.parameters['H']
        
    def observation_covariance(self):
        return self.parameters['R']



class SparseLinearModel(AbstractLinearModel):
    """
    Linear model where the transition matrix is specified as a product of dense
    and sparse components, and the remaining three system matrices are
    specified directly.
    """
    
    def transition_matrix(self):
        return np.mutliply( self.parameters['A'], self.parameters['B'] )

    def transition_covariance(self):
        return self.parameters['Q']
        
    def observation_matrix(self):
        return self.parameters['H']
        
    def observation_covariance(self):
        return self.parameters['R']



class DegenerateLinearModel(AbstractLinearModel):
    """
    Linear model where transition covariance is degenerate and thus
    parameterised in terms of an eigendecomposition. The remaining three
    system matrices are specified directly.
    """
    
    def transition_matrix(self):
        return self.parameters['F']

    def transition_covariance(self):
        eVec = self.parameters['vec']
        eVal = np.diag(self.parameters['val'])
        return np.dot(eVec, np.dot(eVal, eVec.T))
        
    def observation_matrix(self):
        return self.parameters['H']
        
    def observation_covariance(self):
        return self.parameters['R']
    
    def convert_to_givens_form(self):
        """
        Convert from eigen to givens form
        """
        dim,rank = self.parameters['vec'].shape
        Uc,E,Ur = giv.givensise(self.parameters['vec'])
        EUr = np.dot(E[:rank,:rank],Ur)
        D = np.dot(np.dot(EUr, np.diag(self.parameters['val'])), EUr.T)
        U = Uc[:,:rank]
        return U, D
    
    def update_from_givens_form(self, U, D):
        """
        Convert transition covariance from givens to eigen form and update
        """
        eVal, eVec = la.eigh(D)
        self.parameters['vec'] = np.dot(U, eVec)
        self.parameters['val'] = eVal
    
    def complete_basis(self):
        """
        Returns a matrix of vectors orthogonal to the eigenvectors for the
        transition covariance
        """
        Q,R = la.qr(self.parameters['vec'])
        r = self.parameters['rank'][0]
        return Q[:,r:]
    
    def rotate_transition_covariance(self, rotation):
        """
        Rotate transition covariance matrix by multiplying eigenvectors by a
        supplied orthoginal matrix
        """
        self.parameters['vec'] = np.dot(rotation, self.parameters['vec'])
    
    def add_eigen_value_vector(self, value, vector):
        """
        Add an eigenvalue/eigenvector pair to the transition covariance matrix
        """
        if self.parameters['rank'][0] == self.ds:
            raise ValueError("Covariance matrix is already full rank.")
        self.parameters['val'] = np.append(self.parameters['val'], value)
        self.parameters['vec'] = np.append(self.parameters['vec'],
                                           vector[:,np.newaxis], axis=1)
        self.parameters['rank'][0] += 1
    
    def remove_min_eigen_value_vector(self):
        """
        Remove the minimum eigenvalue and the corresponding eigenvector from
        the transition covariance matrix
        """
        minIdx = np.argmin(self.parameters['val'])
        value = self.parameters['val'][minIdx]
        vector = self.parameters['vec'][:,minIdx]
        self.parameters['val'] = np.delete(self.parameters['val'], minIdx)
        self.parameters['vec'] = np.delete(self.parameters['vec'], minIdx,
                                                                       axis=1)
        self.parameters['rank'][0] -= 1
        return value, vector
        
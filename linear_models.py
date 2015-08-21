import numpy as np
from basic import AbstractLinearModel

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
        return np.dot( self.parameters['A'], self.parameters['B'] )

    def transition_covariance(self):
        return self.parameters['Q']
        
    def observation_matrix(self):
        return self.parameters['H']
        
    def observation_covariance(self):
        return self.parameters['R']
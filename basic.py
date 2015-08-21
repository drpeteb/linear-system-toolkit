# Operational modules
from abc import ABCMeta, abstractmethod
import collections

# Numerical modules
import numpy as np
from scipy.stats import multivariate_normal as mvn




# Import Kalman filter/smoother operations
import kalman as kal











#class SparseLinearModel(LinearModel):
#    """Sparse Transition Matrix Linear-Gaussian Model Class"""
#    
#    def __init__(self, P1, A, B, G, H, R):
#        self.A = A.copy()           # Dense transition matrix
#        self.B = B.copy()           # Transition matrix mask
#        self.G = G.copy()           # Transition noise matrix
#        self.H = H.copy()           # Observation matrix
#        self.R = R.copy()           # Observation variance
#        self.P1 = P1.copy()         # First state prior
#        self.ds = A.shape[0]        # State dimension
#        self.do = H.shape[0]        # State dimension
#        
#    def copy(self):
#        return SparseLinearModel(
#                               self.P1, self.A, self.B, self.G, self.H, self.R)
#    
#    @property
#    def F(self):
#        """Sparse Transition Matrix"""
#        return np.multiply(self.A,self.B)
#    @F.setter
#    def F(self, value):
#        raise AttributeError("Can't set attribute")        
#    @F.deleter
#    def F(self):
#        raise AttributeError("Can't delete attribute")
#
#
#
#class DegenerateLinearModel(LinearModel):
#    """Degenerate Transition Covariance Linear-Gaussian Model Class"""
#    
#    def __init__(self, P1, F, D, givens, H, R):
#        self.F = F.copy()           # Transition matrix
#        self.H = H.copy()           # Observation matrix
#        self.R = R.copy()           # Observation variance
#        self.P1 = P1.copy()         # First state prior
#        self.ds = F.shape[0]        # State dimension
#        self.do = H.shape[0]        # State dimension
#        self.rank = D.shape[0]      # Rank of noise
#        
#        self._D = D.copy()              # Postive definite part of transition noise matrix
#        self._givens = tuple(givens)    # Givens rotations for the orthogonal part of transition noise matrix
#        
#        self.order = []                 # Order for givens rotations (building up the product from the right)
#        for ii in range(self.rank):
#            for jj in reversed(range(self.rank,self.ds)):
#                self.order.append((ii,jj))
#        
#        self.U_update()
#        
#    def copy(self):
#        return DegenerateLinearModel(
#             self.P1, self.F, self.D, self.givens, self.H, self.R)
#    
#    @property
#    def G(self):
#        """Transition Noise Matrix"""
#        return self._G
#    @G.setter
#    def G(self, value):
#        raise AttributeError("Can't set attribute")        
#    @G.deleter
#    def G(self):
#        raise AttributeError("Can't delete attribute")
#
#    @property
#    def Q(self):
#        """Transition Covariance Matrix"""
#        return np.dot(self.G,self.G.T)
#    @Q.setter
#    def Q(self, value):
#        raise AttributeError("Can't set attribute")        
#    @Q.deleter
#    def Q(self):
#        raise AttributeError("Can't delete attribute")
#    
#    @property
#    def givens(self):
#        """Givens rotations parameterising orthogonal part of G"""
#        return self._givens
#    @givens.setter
#    def givens(self, value):
#        self._givens = value
#        self.U_update()
#    @givens.deleter
#    def givens(self):
#        raise AttributeError("Can't delete attribute")
#        
#        
#    def set_givens(self, g_ind, new_ga):
#        """Update a single givens rotation"""
#        Ng = len(self._givens)
#        
#        wrap = False
#        if new_ga > np.pi/2:
#            new_ga -= np.pi
#            wrap = True
#        elif new_ga < -np.pi/2:
#            new_ga += np.pi
#            wrap = True
#            
#        self._givens = self._givens[:g_ind] + (new_ga,) + self._givens[g_ind+1:]
#        if len(self._givens)!=Ng:
#            raise ValueError("Number of givens rotations must remain constant")
#        
#        # Orientation correction
#        if wrap:
#            (i_ind,j_ind) = self.order[g_ind]
#            self._D[i_ind,:] *= -1
#            self._D[:,i_ind] *= -1
#            for gg in reversed(range(g_ind)):
#                (ii,jj) = self.order[gg]
#                if len(np.intersect1d( np.array((ii,jj)), np.array((i_ind,j_ind)) ))==1:
#                    self._givens = self._givens[:gg] + (-self._givens[gg],) + self._givens[gg+1:]
#                    
#        self.U_update()
#        
#        return wrap
#        
#    @property
#    def U(self):
#        """Orthogonal part of transition noise matrix"""
#        return self._U        
#    @U.setter
#    def U(self, value):
#        raise AttributeError("Can't set attribute")        
#    @U.deleter
#    def U(self):
#        raise AttributeError("Can't delete attribute")
#    
#    @property
#    def D(self):
#        """Positive definite part of G"""
#        return self._D
#    @D.setter
#    def D(self, value):
#        self._D = value
#        self.G_update()
#    @D.deleter
#    def D(self):
#        raise AttributeError("Can't delete attribute")
#
#    def G_update(self):
#        """Recalculate G from its parameters"""
#        self._G = np.dot( self._U, np.vstack( (self._D, np.zeros((self.ds-self.rank,self.rank))) ) ) 
#    
#    def U_update(self):
#        """Recalculate orthogonal matrix when givens rotations are changed"""
#        M = np.identity(self.ds)
#        gg = 0
#        for (ii,jj) in self.order:
#            rot = np.identity(self.ds)
#            rot[ii,ii] = np.cos(self.givens[gg])
#            rot[jj,jj] = np.cos(self.givens[gg])
#            rot[ii,jj] = np.sin(self.givens[gg])
#            rot[jj,ii] =-np.sin(self.givens[gg])
#            M = np.dot( M, rot)
#            gg += 1
#        self._U = M
#        self.G_update()
#        
#    def givens_prior(self):
#        """Calculate prior probability of a set of givens rotations"""
#        prob = 0
#        for gg in range(len(self.order)):
#            (ii,jj) = self.order[gg]
#            prob += np.log( np.cos(self.givens[gg])**(self.ds-1-jj+ii) )
#        return prob
#    
#    def planar_rotate_noise(self, ii, jj, rot):
#        """Multiply orthogonal component of noise matrix by a planar rotation"""
#        
#        sf = givmat(rot, ii, jj, self.ds)
#        self.rotate_noise(sf)
#        
#        
#    def rotate_noise(self, sf):
#        """Multiply orthogonal component of noise matrix by some input
#           (which should be orthogonal)"""
#        
#        # Eigendecomposition of D and sort into descending order
#        Dval,Dvec = la.eigh(self.D)
#        idx = Dval.argsort()
#        idx = idx[::-1]
#        Dval = Dval[idx]
#        Dvec = Dvec[:,idx]
#        
#        # Find minimal orthogonal matrix
#        U = self.U
#        Uinc = np.identity(self.ds)
#        Uinc[:self.rank,:self.rank] = Dvec
#        U = np.dot(U,Uinc)
#        U = np.delete(U, list(range(self.rank,self.ds)), 1)
#        
#        # Apply the scaling
#        Unew = np.dot(sf, U)
#        
#        # Factorise orthogonal matrix back into Givens rotations
#        Ur,Uc,givens,order,E = givensise(Unew)
#        self.order = order
#        self._givens = tuple(givens)
#        
#        # Rebuild D
#        EUr = np.dot(E[:self.rank,:self.rank],Ur)
#        self._D = np.dot(EUr,np.dot(np.diag(Dval),EUr.T))
#        
#        self.U_update()
#        
#    
#    def reduce_rank(self, scale):
#        """Reduce the rank of Q/G and modify givens rotations and D
#        accordingly"""
#        
#        # Eigendecomposition of D and sort into descending order
#        Dval,Dvec = la.eigh(self.D)
#        idx = Dval.argsort()
#        idx = idx[::-1]
#        Dval = Dval[idx]
#        Dvec = Dvec[:,idx]
#        
#        # Find minimal orthogonal matrix
#        U = self.U
#        Uinc = np.identity(self.ds)
#        Uinc[:self.rank,:self.rank] = Dvec
#        U = np.dot(U,Uinc)
#        U = np.delete(U, list(range(self.rank,self.ds)), 1)
#        
#        # Reduce rank
#        self.rank -= 1
#        
#        # Remove an eigenvalue and the corresponding eigenvector
#        discard_val = Dval[self.rank]
#        Dval = np.delete(Dval, self.rank)
#        discard_vec = U[:,self.rank]
#        U = np.delete(U, self.rank, 1)
#        
#        # Factorise orthogonal matrix back into Givens rotations
#        Ur,Uc,givens,order,E = givensise(U)
#        self.order = order
#        self._givens = tuple(givens)
#        
#        # Rebuild D
#        EUr = np.dot(E[:self.rank,:self.rank],Ur)
#        self._D = np.dot(EUr,np.dot(np.diag(Dval),EUr.T))
#        
##        Dfull = np.zeros((self.ds,self.ds))
##        Dfull[:self.rank,:self.rank] = np.dot(self._D,self._D)
##        print(np.dot(Uc,np.dot(Dfull,Uc.T)))
#        
#        self.U_update()
#        
#        # Dimension correction factor
##        dcf = 0
#        r = self.rank
#        second_val = 1./(Dval[r-1]**2)
#        dcf = np.log( np.sqrt(np.pi) ) \
#             +np.log( sps.gammaincc((self.ds-r)/2, second_val/(2*scale)) ) \
#             -(r/2)*np.log(2*scale) \
#             -sps.gammaln((r+1)/2) \
#             -0.5*np.sum(np.log(1./(Dval**2))+np.log(1./(discard_val**2)-1./(Dval**2)))
#        dcf = -dcf
#        
#        return dcf,discard_val,discard_vec
#
#    def increase_rank(self, new_val, new_vec, scale):
#        """Increase the rank of Q/G and modify givens rotations and D
#        accordingly"""
#        
#        # Note, new_vec is assumed to be valid, i.e. orthogonal to the existing
#        # eigenvectors of Q
#        
#        # Eigendecomposition of D and sort into descending order
#        Dval,Dvec = la.eigh(self.D)
#        idx = Dval.argsort()
#        idx = idx[::-1]
#        Dval = Dval[idx]
#        Dvec = Dvec[:,idx]
#        
#        # Find minimal orthogonal matrix
#        U = self.U
#        Uinc = np.identity(self.ds)
#        Uinc[:self.rank,:self.rank] = Dvec
#        U = np.dot(U,Uinc)
#        U = np.delete(U, list(range(self.rank,self.ds)), 1)
#        
#        # If no new eigenvector is supplied, sample from uniform
#        new_vec = mvn.rvs(mean=np.zeros(self.ds),cov=np.identity(self.ds))
#        eigs = np.append(U,np.expand_dims(new_vec,1),1)
#        orth,_ = la.qr(eigs,mode='economic')
#        new_vec = orth[:,-1]
#        
#        # If no new eigenvalue is supplied, sample sensible proposal
#        shape = (self.ds-self.rank)/2
#        lowerbound = 1./(Dval[-1]**2)
#        val = sample_truncated_gamma(shape,scale,lowerbound)
#        new_val = 1./np.sqrt(val)
#        
##        print(np.dot(U.T,new_vec))
#        
#        # Increase rank
#        self.rank += 1
#        
#        # Add new eigenvalue and vector
#        Dval = np.append(Dval, [new_val])
#        U = np.append(U, np.expand_dims(new_vec,1), 1)
#        
##        print(np.dot(U,np.dot(np.diag(Dval**2),U.T)))
##        print('')
#        
#        # Factorise orthogonal matrix back into Givens rotations
#        Ur,Uc,givens,order,E = givensise(U)
#        self.order = order
#        self._givens = tuple(givens)
#        
#        # Rebuild D
#        EUr = np.dot(E[:self.rank,:self.rank],Ur)
#        self._D = np.dot(EUr,np.dot(np.diag(Dval),EUr.T))
#        
##        Dfull = np.zeros((self.ds,self.ds))
##        Dfull[:self.rank,:self.rank] = np.dot(self._D,self._D)
##        print(np.dot(Uc,np.dot(Dfull,Uc.T)))
#        
#        self.U_update()
#        
#        # Dimension correction factor
##        dcf = 0
#        r = self.rank - 1
#        second_val = 1./(Dval[r-1]**2)
#        dcf = np.log( np.sqrt(np.pi) ) \
#             +np.log( sps.gammaincc((self.ds-r)/2, second_val/(2*scale)) ) \
#             -(r/2)*np.log(2*scale) \
#             -sps.gammaln((r+1)/2) \
#             -0.5*np.sum(np.log(1./(Dval[:-1]**2))+np.log(1./(new_val**2)-1./(Dval[:-1]**2)))
#             
#        return dcf
#              
#        
#
#def planerot(x):
#    """Find the plane rotation and Givens angle for to zero a component"""
#    if x[1] != 0:
#        r = la.norm(x)
#        G = np.hstack((np.expand_dims(x,1),np.expand_dims(x[::-1],1)))/r
#        G[1,0] *= -1
#        y = np.array([r,0])
#    else:
#        G = np.identity(2)
#        y = x
##    g = np.arctan2(G[0,1],G[0,0])
#    g = np.arctan(G[0,1]/G[0,0])
#    return y,g,G
#    
#def givmat(g,ii,jj,d):
#    """create a givens rotation matrix"""
#    G = np.identity(d)
#    G[ii,ii] = np.cos(g)
#    G[jj,jj] = np.cos(g)
#    G[ii,jj] = np.sin(g)
#    G[jj,ii] =-np.sin(g)
#    return G
#    
#def givensise(U):
#    """Factorise an orthogonal matrix into row space and cross rotations, and a
#    sign matrix, such that U = Ur x E x Uc"""
#    
#    U_original = U.copy()
#    d,r = U.shape
#    
#    Ur = np.identity(r)
#    Uc = np.identity(d)
#    
#    # Row space loop
#    for rr in reversed(range(r)):
#        for cc in range(rr):
#            
#            v = U[rr,[rr,cc]]
#            _,g,_ = planerot(v)
##            G = np.identity(d)
##            G[[cc,rr],[cc,rr]] = Gs
#            G = givmat(g,cc,rr,r)
#            # test that U[rr]*G=unit vector
#            U = np.dot(U,G);
#            Ur = np.dot(G.T,Ur)
#            
#    # Cross loop
#    givens = []
#    order = []
#    for cc in range(r):
#        for rr in reversed(range(r,d)):
#            
#            v = U[[cc,rr],cc]
#            _,g,_ = planerot(v)
#            G = givmat(g,cc,rr,d)
#            U = np.dot(G,U)
#            Uc = np.dot(Uc,G.T)
#            
#            givens.append(-g)
#            order.append((cc,rr))
#    
#    E = U
##    E = np.identity(d)
##    E = np.delete(E,range(r,d),1)
#    
#    if not np.allclose(U_original, np.dot(Uc,np.dot(E,Ur)) ):
#        raise ValueError("Givens factorisation failed")
#    
##    order = order[::-1]
#    
##    print(U)
##    print('')
##    print(U_original)
##    print('')
##    print(np.dot(Uc,np.dot(E,Ur)))
#    
#    return Ur,Uc,givens,order,E
#    
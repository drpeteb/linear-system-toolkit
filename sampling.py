import numpy as np
from scipy import linalg as la
from scipy import stats
from scipy import special
#from scipy.stats import multivariate_normal as mvn
#from scipy import stats as sps
#from scipy import misc as spm
#from numpy import random as rnd

def sample_wishart(nu, P):
    """
    Sample from a Wishart distribution.
    
    N.B. There is a function in scipy 0.17, the development version,
    but not the current 0.16. Probably worth updating when it gets released.
    """
    dim = P.shape[0]
    cholP = la.cholesky(P,lower=True)
    R = np.zeros((dim,dim))
    for ii in range(dim):
        R[ii,ii] = np.sqrt(stats.chi2.rvs(nu-(ii+1)+1))
        for jj in range(ii+1,dim):
            R[jj,ii] = stats.norm.rvs()
    cholX = np.dot(cholP,R)
    X = np.dot(cholX,cholX.T)
    return X

def sample_matrix_normal(M, U, V):
    """Sample from a matrix normal distribution"""
    Mv = M.T.flatten()
    Sv = np.kron(V,U)
    Xv = stats.multivariate_normal.rvs(mean=Mv,cov=Sv)
    X = np.reshape(Xv,M.T.shape).T
    return X

def sample_cayley(d, s):
    """
    Sample a random orthogonal matrix using normal variates and the Cayley
    transformation
    """
    
    # Random skew-symmetric matrix
    S = np.zeros((d,d))
    for dd in range(d-1):
        y = stats.norm.rvs(loc=np.zeros(d-dd-1), scale=s*np.ones(d-dd-1))
        S[dd,dd+1:] = y
        S[dd+1:,dd] = -y
    
    # Cayley transformation
    I = np.identity(d)
    M = la.solve(I-S,I+S)
    
    return M

def singular_wishart_density(val, vec, P):
    """
    Density of a singular wishart distribution at X = vec*diag(val)*vec.T
    """
    d,r = vec.shape
    norm = 0.5*r*(r-d)*np.log(np.pi) - 0.5*r*d*np.log(2.0) \
           - special.multigammaln(r/2,r) - 0.5*r*la.det(P)
    pptn = 0.5*(r-d-1)*np.log(np.prod(val)) \
           - 0.5*np.trace( np.dot(np.dot(vec.T,la.solve(P,vec)),np.diag(val)) )
    pdf = norm + pptn
    return pdf

def matrix_normal_density(X, M, U, V):
    """Sample from a matrix normal distribution"""
    norm = - 0.5*la.det(2*np.pi*U) - 0.5*la.det(2*np.pi*V)
    XM = X-M
    pptn = np.exp(-0.5*np.trace( np.dot(la.solve(U,XM),la.solve(V,XM.T)) ))
    pdf = norm + pptn
    return pdf

def evaluate_sufficient_statistics(x):
    """
    Calculate the required sufficient statistics from the state trajectory
    """
    K,ds = x.shape
    
    # Create arrays
    suffStats = list()
    suffStats.append( 0 )
    suffStats.append( np.zeros((ds,ds)) )
    suffStats.append( np.zeros((ds,ds)) )
    suffStats.append( np.zeros((ds,ds)) )
    
    # Loop through time incrementing stats
    for kk in range(1,K):
        suffStats[0] += 1
        suffStats[1] += np.outer(x[kk-1], x[kk-1])
        suffStats[2] += np.outer(x[kk], x[kk-1])
        suffStats[3] += np.outer(x[kk], x[kk])
    
    return suffStats
    


def sample_basic_transition_mniw_conditional(suffStats, nu0, Psi0, M0, V0):
    """
    Sample transition model matrices from matrix-normal-inverse-wishart
    posterior conditional distribution.
    """
    invV0 = la.inv(V0)
    
    # Posterior hyperparameters    
    nu  = nu0 + suffStats[0]
    V   = la.inv( invV0 + suffStats[1] )
    M   = np.dot( np.dot(M0,invV0)+suffStats[2] , V)
    Psi = Psi0 + suffStats[3] - np.dot(M, la.solve(V,M.T) ) + np.dot(M0, la.solve(V0,M0.T) )
    
    # Sample
    Q = la.inv(sample_wishart(nu, la.inv(Psi)))
    F = sample_matrix_normal(M, Q, V)

    return F, Q

def sample_basic_transition_matrix_mniw_conditional(suffStats, Q, M0, V0, F=None, with_pdf=False):
    """
    Sample transition matrix from matrix-normal posterior conditional
    distribution.
    """
    invV0 = la.inv(V0)
    
    # Posterior hyperparameters    
    V   = la.inv( invV0 + suffStats[1] )
    M   = np.dot( np.dot(M0,invV0)+suffStats[2] , V)
    
    # Sample
    if F is None:
        F = sample_matrix_normal(M, Q, V)

    if not with_pdf:
        return F
    else:
        pdf = matrix_normal_density(F, M, Q, V)
        return F, pdf



def sample_degenerate_transition_mniw_conditional(suffStats, U, Fold, nu0, Psi0, M0, V0):
    """
    Sample transition model matrices from singular 
    matrix-normal-inverse-wishart posterior conditional distribution.
    """
    invV0 = la.inv(V0)
    ds = V0.shape[0]
    
    # Posterior hyperparameters    
    nu  = nu0 + suffStats[0]
    V   = la.inv( invV0 + suffStats[1] )
    M   = np.dot( np.dot(U.T, np.dot(M0,invV0)+suffStats[2]) , V)
    UPsiU = Psi0+suffStats[3]+np.dot(M0,la.solve(V0,M0.T))
    Psi = np.dot(U.T, np.dot(UPsiU, U)) - np.dot(M, la.solve(V,M.T))
    
    # Sample
    D = sample_wishart(nu, la.inv(Psi))
    FU = sample_matrix_normal(M, D, V)
    
    # Project back out
    F = np.dot( (np.identity(ds)-np.dot(U,U.T)), Fold ) + np.dot(U,FU)

    return F, D










#def sample_transition_model_prior(nu0, Psi0, M0, V0, A=None, Q=None):
#    """Sample transition model from matrix-normal-inverse-wishart prior"""
#        
#    if Q is None:
#        Q = la.inv(sample_wishart(nu0, la.inv(Psi0)))
#    
#    if A is None:
#        A = sample_matrix_normal(M0, Q, V0)
#
#    return A, Q
#
#def sample_transition_model_conditional(x, nu0, Psi0, M0, V0, A=None, Q=None):
#    """Sample transition model from matrix-normal-inverse-wishart posterior"""
#    
#    ds = M0.shape[0]
#    K = len(x)
#    invV0 = la.inv(V0)
#    
#    # Sufficient stats
#    SS0 = 0
#    SS1 = np.zeros((ds,ds))
#    SS2 = np.zeros((ds,ds))
#    SS3 = np.zeros((ds,ds))
#    for kk in range(1,K):
#        SS0 += 1
#        SS1 += np.outer(x[kk-1], x[kk-1])
#        SS2 += np.outer(x[kk-1], x[kk])
#        SS3 += np.outer(x[kk], x[kk])
#    
#    nu  = nu0 + SS0
#    V   = la.inv( invV0 + SS1 )
#    M   = np.dot( np.dot(M0,invV0)+SS2.T , V)
#    Psi = Psi0 + SS3 - np.dot(M, la.solve(V,M.T) ) + np.dot(M0, la.solve(V0,M0.T) )
#    
#    
#    if Q is None:
#        Q = la.inv(sample_wishart(nu, la.inv(Psi)))
#    
#    if A is None:
#        A = sample_matrix_normal(M, Q, V)
#        
##    print(A)
#
#    return A, Q
#
#
#### PRIORS ###
#
#def sample_transition_matrix_prior(ds, pA, A=None):
#    """Sample a transition matrix from a Gaussian prior (mean 0,
#       covariance pA*I) over its vectorised elements"""
#    ds2 = ds**2
#    P = pA*np.identity(ds2)
#    if not(type(A)==np.ndarray):
#        Avec = mvn.rvs(mean=np.zeros(ds2), cov=P)
#        A = np.reshape(Avec,(ds,ds))
#    else:
#        Avec = A.flatten()
#    pdf = mvn.logpdf(Avec,mean=np.zeros(ds2),cov=P)
#    return A,pdf
#    
#def sample_transition_matrix_mask_prior(ds, pB):
#    """Sample a transition matrix mask from an independent Bernoulli prior 
#       over its elevemnts"""
#    B = sps.bernoulli.rvs(pB, size=(ds,ds))
#    return B
#
#def sample_transition_covariance_prior(ds, nu0, pQ, Q=None):
#    """Sample a transition covariance matrix from its inverse Wishart prior"""
#    P = pQ*np.identity(ds)
#    if not(type(Q)==np.ndarray):
#        invQ = sample_wishart(nu0,P)
#        Q = la.inv(invQ)
#    else:
#        invQ = la.inv(Q)
#    pdf = wishart_density(invQ,nu0,P)
#    return Q,pdf
#

#def sample_observation_covariance_scale_conditional(x, y, H, nu0, pR):
#    
#    shape = nu0
#    rate = pR
#    for kk in range(len(y)):
#        if len(y[kk])>0:
#            shape += len(y[kk])
#            rate += la.norm( y[kk]-np.dot(H[kk],x[kk]) )**2
#    Rinvscale = stats.gamma.rvs(shape,scale=1./rate)
#    Rscale = 1./Rinvscale
#    return Rscale
#    
#
#### SPARSE LINEAR MODEL CONDTIONALS ###
#
#def sample_transition_matrix_mask_conditional(x, A, B, Q, pB):
#    """Sample a transition matrix mask from its posterior conditional on the
#       sampled state sequence"""
#    K = len(x)
#    ds = A.shape[1]
#    ds2 = pow(ds,2)
#    XQX = np.zeros((ds2,ds2))
#    XQx = np.zeros(ds2)
#    for kk in range(1,K):
#        tmp = [ np.multiply((A[ii,:]), x[kk-1]) for ii in range(ds) ]
#        Xmat = la.block_diag(*tmp)
#        XQX = XQX + np.dot( la.solve(Q,Xmat,sym_pos=True,check_finite=False).T ,Xmat )
#        XQx = XQx + np.dot( la.solve(Q,Xmat,sym_pos=True,check_finite=False).T ,x[kk])
#    Bvec = B.flatten()
#    order = np.random.permutation(ds2)
#    pon = np.zeros(2)
#    for ii in order:
#        Bmod = list(Bvec)
#        Bmod[ii] = 0.5
#        pon[0] = np.log(pB) - 0.5*(  2*np.dot(XQX[ii,:],Bmod)-2*XQx[ii]  )
#        pon[1] = np.log(1-pB)
#        pon = pon - spm.logsumexp(pon)
#        Bvec[ii] = np.log(rnd.random())<pon[0]
#    B = np.reshape(Bvec,(ds,ds))
#    return B
#    
#def sample_transition_matrix_and_mask_conditional(x, A, B, Q, pA, pB):
#    """Sample a transition matrix AND its mask jointly element-wise from 
#       their posterior conditional on the sampled state sequence"""    
#    K = len(x)
#    ds = A.shape[0]
#    
#    # Don't change originals
#    Anew = A.copy()
#    Bnew = B.copy()
#    
#    order1 = np.random.permutation(ds)
#    order2 = np.random.permutation(ds)
#    
#    for ii in order1:
#        for jj in order2:
#            XQx = 0
#            xQx = 0
#            F = np.multiply(Anew,Bnew)
#            F[ii,jj] = 0
#            for kk in range(1,K):
#                xij = np.zeros(ds)
#                xij[ii] = x[kk-1][jj]
#                D = x[kk]-np.dot(F,x[kk-1])
#                XQx = XQx + np.dot(D.T, la.solve(Q,xij) )
#                xQx = xQx + np.dot(xij.T, la.solve(Q,xij) )
#            
#            pon = np.zeros(2)
#            vr = 1/pA + xQx
#            pon[0] = np.log(pB) + 0.5*( (XQx**2)/vr ) - 0.5*np.log(vr)
#            pon[1] = np.log(1-pB) - 0.5*np.log(1/pA)
#            pon = pon - spm.logsumexp(pon)
#            Bnew[ii,jj] = np.log(rnd.random())<pon[0]
#            
#            if Bnew[ii,jj]:
#                a_vr = 1/(1/pA + xQx)
#                a_mn = a_vr*XQx
#            else:
#                a_vr = pA
#                a_mn = 0
#            Anew[ii,jj] = mvn.rvs(mean=a_mn, cov=a_vr)
#            
#    return Anew,Bnew
#

#    
#def sample_truncated_gamma(a,b,c):
#    """Sample from a truncated gamma distribution"""
#    lb = stats.gamma.cdf(c,a,scale=b)
#    u = stats.uniform.rvs(loc=lb,scale=1-lb)
#    x = stats.gamma.ppf(u,a,scale=b)
#    return x
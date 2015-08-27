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

def singular_wishart_density(val,vec,P):
    """
    Density of a singular wishart distribution at X = vec*diag(val)*vec.T
    """
    d,r = vec.shape
    norm = 0.5*r*(r-d)*np.log(np.pi) - 0.5*r*d*np.log(2.0) \
           - special.multigammaln(r/2,r) - 0.5*r*la.det(P)
    prptn = (r-d-1)*np.log(np.prod(val)) \
           - 0.5*np.trace( np.dot(np.dot(vec.T,la.solve(P,vec)),np.diag(val)) )
    pdf = norm + prptn
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







def sample_basic_transition_matrix_independent_conditional(suffStats, M0, 
                                                                     alpha, Q):
    """
    Sample transition matrix from posterior conditional distribution arising 
    from isotropic matrix normal prior
    """
    ds = Q.shape[0]
    
    # Eigendecompositions
    QeVal,QeVec = la.eigh(Q)
    ss1eVal, ss1eVec = la.eigh(suffStats[1])
    
    # Hyperparameters of rotated transition matrix
    ss2rot = np.dot(np.dot(QeVec.T, suffStats[2]), ss1eVec)
    M0rot = np.dot(np.dot(QeVec.T, M0), ss1eVec)
    P = 1./( 1./alpha + np.kron(1./QeVal, ss1eVal) )
    m = P*( (np.dot(np.diag(1./QeVal), ss2rot)+M0rot/alpha).flatten() )
    
    # Sample rotated, flattened transition matrix
    Frotflat = stats.norm.rvs(loc=m,scale=np.sqrt(P))
    
    # Reshape into a matrix
    Frot = Frotflat.reshape((ds,ds))
    
    # Rotate to get the final transition matrix
    F = np.dot(np.dot(QeVec, Frot), ss1eVec.T)
    
    return F


def sample_basic_transition_covariance_independent_conditional(suffStats, nu0,
                                                                      Psi0, F):
    """
    Sample transition covariance from posterior conditional distribution
    arising from inverse wishart prior
    """
    
    # Update covariance hyperparameters
    nu  = nu0 + suffStats[0]
    Psi = Psi0 + suffStats[3] - np.dot(F,suffStats[2].T) \
                - np.dot(suffStats[2],F.T) + np.dot(np.dot(F,suffStats[1]),F.T)
    
    # Sample
    Q = la.inv(sample_wishart(nu, la.inv(Psi)))
    
    return Q
    





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
#### BASIC LINEAR MODEL CONDTIONALS ###
#
#def sample_transition_matrix_conditional(x, Q, pA, B=None, A=None):
#    """Sample a transition matrix from its posterior conditional on the
#       sampled state sequence. Also return the pdf of the sampled matrix."""
#    K = len(x)
#    ds = Q.shape[0]
#    ds2 = ds**2
#    P = pA*np.identity(ds2)
#    XQX = np.zeros((ds2,ds2))
#    XQx = np.zeros(ds2)
#    for kk in range(1,K):
#        if type(B)==np.ndarray:
#            tmp = [ np.multiply((B[ii,:]), x[kk-1]) for ii in range(ds) ]
#        else:
#            tmp = [x[kk-1]]*ds
#        Xmat = la.block_diag(*tmp)
#        XQX = XQX + np.dot( la.solve(Q,Xmat,sym_pos=True,check_finite=False).T ,Xmat )
#        XQx = XQx + np.dot( la.solve(Q,Xmat,sym_pos=True,check_finite=False).T ,x[kk])
#    Avr = la.inv(la.inv(P, check_finite=False) + XQX, check_finite=False)
#    Avr = (Avr+Avr.T)/2
#    Amn = np.dot(Avr,XQx)
#    if not(type(A)==np.ndarray):
#        Avec = mvn.rvs(mean=Amn, cov=Avr)
#        A = np.reshape(Avec,(ds,ds))
#    else:
#        Avec = A.flatten()
#    pdf = mvn.logpdf(Avec,mean=Amn,cov=Avr)
#    return A,pdf
#    
#def sample_transition_covariance_conditional(x, F, nu0, pQ, Q=None):
#    """Sample a transition covariance matrix from its posterior conditional
#       on the sampled state sequence"""
#    K = len(x)
#    ds = F.shape[0]
#    invP = np.identity(ds)/pQ
#    for kk in range(1,K):
#        delta = x[kk]-np.dot(F,x[kk-1])
#        invP = invP + np.outer(delta,delta)
#    nu = nu0 + K-1
#    P = la.inv(invP)
#    if not(type(Q)==np.ndarray):
#        invQ = sample_wishart(nu,P)
#        Q = la.inv(invQ)
#    else:
#        invQ = la.inv(Q)
#    pdf = wishart_density(invQ,nu,P)
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
#### DEGENERATE LINEAR MODEL CONDTIONALS ###
#
#def sample_degenerate_transition_matrix_conditional(x, A, Q, Om, pA):
#    """Sample a transition matrix from its posterior conditional on the
#       sampled state sequence when the noise covariance matrix is degenerate"""       
#    K = len(x)
#    ds = A.shape[0]
#    r = Q.shape[0]
#    
#    z = x.copy()
#    z[0] = np.zeros(z[0].shape)
#    for kk in range(1,K):
#        z[kk] = np.dot(Om.T, x[kk]-np.dot(A,x[kk-1]))
#        if not all(abs(z[kk][r:])<(1E-5*K)):
#            print(z[kk])
#            raise ValueError("State difference in incorrect subspace.")
#    
#    XQX = np.zeros((ds*r,ds*r))
#    XQx = np.zeros(ds*r)
#    for kk in range(1,K):
#        tmp = [x[kk-1]]*r
#        Xmat = la.block_diag(*tmp)
#        XQX = XQX + np.dot( la.solve(Q,Xmat,sym_pos=True,check_finite=False).T ,Xmat )
#        XQx = XQx + np.dot( la.solve(Q,Xmat,sym_pos=True,check_finite=False).T ,z[kk][:r])
#    P = pA*np.identity(r*ds)
#    m = np.dot(-Om[:,:r].T,A).flatten()
#    Avr = la.inv(la.inv(P, check_finite=False) + XQX, check_finite=False)
#    Avr = (Avr+Avr.T)/2
#    Amn = np.dot(Avr,XQx + la.solve(P,m,sym_pos=True,check_finite=False))
#    Avec = mvn.rvs(mean=Amn, cov=Avr)
#    A = A + np.dot(Om, np.vstack((np.reshape(Avec,(r,ds)),np.zeros((ds-r,ds)))) )
#    return A
#
#def sample_degenerate_transition_noise_matrix_conditional(x, F, r, U, nu0, P0):
#    """Sample a degenerate transition noise matrix from its posterior
#       conditional on the sampled state sequence"""
#    K = len(x)
#    delta = [np.dot(U.T,x[nn]-np.dot(F,x[nn-1]))[:r] for nn in range(1,K)]
#    invP = np.identity(r)/P0
#    for kk in range(K-1):
#        invP = invP + np.outer(delta[kk],delta[kk])
#    nu = nu0 + K-1
#    P = la.inv(invP)
#    invQ = sample_wishart(nu,P)
#    Q = la.inv(invQ)
#    D = la.sqrtm(Q)    
#    return D
#    
#### ANNEALED CHANGEPOINT SYSTEM LEARNING ###
#def sample_annealed_transition_matrix_conditional(x, idx1, idx2, Q, pA, gamma):
#    K = len(x)
#    ds = Q.shape[1]
#    ds2 = pow(ds,2)
#    P = pA*np.identity(ds2)
#    XQX = np.zeros((ds2,ds2))
#    XQx = np.zeros(ds2)
#    for kk in range(1,K):
#        if (kk in idx1) and (kk in idx2):
#            Qtemp = Q
#        elif (kk in idx1):
#            Qtemp = Q/(1-gamma)
#        elif (kk in idx2):
#            Qtemp = Q/gamma
#        else:
#            continue
#        tmp = [x[kk-1]]*ds
#        Xmat = la.block_diag(*tmp)
#        XQX = XQX + np.dot( la.solve(Qtemp,Xmat,sym_pos=True,check_finite=False).T ,Xmat )
#        XQx = XQx + np.dot( la.solve(Qtemp,Xmat,sym_pos=True,check_finite=False).T ,x[kk])
#    Avr = la.inv(la.inv(P, check_finite=False) + XQX, check_finite=False)
#    Avr = (Avr+Avr.T)/2
#    Amn = np.dot(Avr,XQx)
#    Avec = mvn.rvs(mean=Amn, cov=Avr)
#    A = np.reshape(Avec,(ds,ds))
#    return A
#    
#def sample_annealed_transition_covariance_conditional(x, idx1, idx2, F, nu0, pQ, gamma):
#    K = len(x)
#    ds = F.shape[0]
#    invP = np.identity(ds)/pQ
#    nu = nu0
#    for kk in range(1,K):
#        delta = x[kk]-np.dot(F,x[kk-1])
#        if (kk in idx1) and (kk in idx2):
#            invP = invP + np.outer(delta,delta)
#            nu += 1
#        elif (kk in idx1):
#            invP = invP + np.outer(delta,delta)/(1-gamma)
#            nu += 1./(1-gamma)
#        elif (kk in idx2):
#            invP = invP + np.outer(delta,delta)/gamma
#            nu += 1./gamma
#        else:
#            continue
#    P = la.inv(invP)
#    invQ = sample_wishart(nu,P)
#    Q = la.inv(invQ)
#    return Q
#    
#
#### BASIC SAMPLING OPERATIONS ###
#
#def sample_wishart(nu, P):
#    dim = P.shape[0]
#    cholP = la.cholesky(P,lower=True)
#    R = np.zeros((dim,dim))
#    for ii in range(dim):
#        R[ii,ii] = np.sqrt(sps.chi2.rvs(nu-(ii+1)+1))
#        for jj in range(ii+1,dim):
#            R[jj,ii] = sps.norm.rvs()
#    cholX = np.dot(cholP,R)
#    X = np.dot(cholX,cholX.T)
#    return X
#    
#def wishart_density(X,nu,P):
#    d = X.shape[0]
#    pdf = 0.5*(nu-d-1)*np.log(la.det(X)) - 0.5*np.trace(la.solve(P,X)) \
#         -0.5*d*nu*np.log(2) - 0.5*nu*np.log(la.det(P)) - spec.multigammaln(nu/2,d)
#    return pdf
#
#def sample_cayley(d, s):
#    """Sample a random Cayley-distributed orthogonal matrix"""
#    
#    # Random skew-symmetric matrix
#    S = np.zeros((d,d))
#    for dd in range(d-1):
#        y = mvn.rvs(mean=np.zeros(d-dd-1), cov=s**2*np.identity(d-dd-1))
#        S[dd,dd+1:] = y
#        S[dd+1:,dd] = -y
#    
#    # Cayley transformation
#    I = np.identity(d)
#    M = la.solve(I-S,I+S)
#    
#    return M
#    
##    # Scalar case
##    if d == 1:
##        M = 1
##        return M
##    
##    # Initialise in 2 dimensions
##    y = stats.t.rvs(k)
##    S
##    
##    # Loop up throu
##    for dd in range(d):
##        
##    
##    return M
#    
#    
#def sample_truncated_gamma(a,b,c):
#    """Sample from a truncated gamma distribution"""
#    lb = stats.gamma.cdf(c,a,scale=b)
#    u = stats.uniform.rvs(loc=lb,scale=1-lb)
#    x = stats.gamma.ppf(u,a,scale=b)
#    return x
import numpy as np
from scipy import linalg as la
from scipy import stats
from scipy import special

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

def sample_orthogonal_haar(d):
    """
    Sample a unit vector uniformly from the (d-1)-sphere
    """
    z = stats.multivariate_normal.rvs(mean=np.zeros(d),cov=np.identity(d))
    z /= la.norm(z)
    if d == 1:
        z = z[np.newaxis] # This undoes the squeeze if only 1D
    return z

def singular_wishart_density(val, vec, P):
    """
    Density of a singular wishart distribution at X = vec*diag(val)*vec.T
    """
    d,r = vec.shape
    norm = 0.5*r*(r-d)*np.log(np.pi) - 0.5*r*d*np.log(2.0) \
          - special.multigammaln(r/2,r) - 0.5*r*np.log(la.det(P))
    pptn = 0.5*(r-d-1)*np.sum(np.log(val)) \
          - 0.5*np.trace( np.dot(np.dot(vec.T,la.solve(P,vec)),np.diag(val)) )
    pdf = norm + pptn
    return pdf

def singular_inverse_wishart_density(val, vec, P):
    """
    Density of a singular inverse wishart distribution at
    X = vec*diag(val)*vec.T
    """
    d,r = vec.shape

    norm = 0.5*r*np.log(la.det(P)) \
            -0.5*r*d*np.log(2.0) \
            -0.5*r*(d-r)*np.log(np.pi) \
            -special.multigammaln(r/2,r)

    pptn = -0.5*(3*d-r+1)*np.sum(np.log(val)) \
           -0.5*np.trace( np.dot(np.dot(vec.T,np.dot(P,vec)),np.diag(1/val)) )
    
    pdf = norm + pptn
    return pdf

def matrix_normal_density(X, M, U, V):
    """Sample from a matrix normal distribution"""
    norm = - 0.5*np.log(la.det(2*np.pi*U)) - 0.5*np.log(la.det(2*np.pi*V))
    XM = X-M
    pptn = -0.5*np.trace( np.dot(la.solve(U,XM),la.solve(V,XM.T)) )
    pdf = norm + pptn
    return pdf



def evaluate_transition_sufficient_statistics(x):
    """
    Calculate the required sufficient statistics from the state trajectory
    for the transition model sampling operations.
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

def evaluate_observation_sufficient_statistics(x, y):
    """
    Calculate the required sufficient statistics from the state trajectory
    for the observation model sampling operations.
    """
    K,ds = x.shape
    _,do = y.shape

    # Create arrays
    suffStats = list()
    suffStats.append( 0 )
    suffStats.append( np.zeros((ds,ds)) )
    suffStats.append( np.zeros((do,ds)) )
    suffStats.append( np.zeros((do,do)) )

    # Loop through time incrementing stats
    for kk in range(K):
        if not np.any(np.isnan(y[kk])):
            suffStats[0] += 1
            suffStats[1] += np.outer(x[kk], x[kk])
            suffStats[2] += np.outer(y[kk], x[kk])
            suffStats[3] += np.outer(y[kk], y[kk])

    return suffStats



def hyperparam_update_basic_mniw_transition(suffStats, nu0, Psi0, M0, V0):
    """
    Update matrix-normal-inverse-wishart hyperparameters for transition model
    matrices conditional on observed state trajectory.
    """

    # Posterior hyperparameters
    nu  = nu0 + suffStats[0]
    V   = la.inv( la.inv(V0) + suffStats[1] )
    M   = np.dot( la.solve(V0,M0.T).T + suffStats[2] , V)
    Psi = Psi0 + suffStats[3] - np.dot(M, la.solve(V,M.T) ) \
                                              + np.dot(M0, la.solve(V0,M0.T) )

    return nu, Psi, M, V

def hyperparam_update_basic_ig_observation_variance(suffStats, H, a0, b0):
    """
    Update inverse-gamma hyperparameters for the scale factor on the
    observation covariance (which is assumed to be diagonal).
    """
    dy = suffStats[3].shape[0]
    Hss2 = np.dot(H,suffStats[2].T)
    sumSquares = suffStats[3] - Hss2 - Hss2.T \
                                          + np.dot(np.dot(H,suffStats[1]),H.T)

    # Posterior hyperparameters
    a = a0 + 0.5*dy*suffStats[0]
    b = b0 + 0.5*np.trace( sumSquares )

    return a,b

def hyperparam_update_basic_mn_transition_matrix(suffStats, M0, V0):
    """
    Update matrix-normal hyperparameters for transition matrix conditional
    on observed state trajectory.
    """

    # Posterior hyperparameters
    V = la.inv( la.inv(V0) + suffStats[1] )
    M = np.dot( la.solve(V0,M0.T).T + suffStats[2] , V)

    return M, V

def hyperparam_update_basic_iw_transition_covariance(suffStats, F, nu0, Psi0):
    """
    Update inverse-wishart hyperparameters for transition covariance matrix
    conditional on observed state trajectory.
    """

    # Posterior hyperparameters
    Fss2 = np.dot(F,suffStats[2].T)
    nu = nu0 + suffStats[0]
    Psi = Psi0 + suffStats[3] - Fss2 - Fss2.T \
                                          + np.dot(np.dot(F,suffStats[1]),F.T)

    return nu, Psi

def hyperparam_update_degenerate_mniw_transition(suffStats, U, nu0, Psi0,
                                                                      M0, V0):
    """
    Update matrix-normal-inverse-wishart hyperparameters for within-subspace
    comonents of the transition model matrices conditional on observed state
    trajectory.
    """

    # Posterior hyperparameters
    nu = nu0 + suffStats[0]
    V = la.inv( la.inv(V0) + suffStats[1] )
    M = np.dot( np.dot(U.T, la.solve(V0,M0.T).T + suffStats[2]) , V)
    UPsi0U = la.inv(np.dot(U.T,la.solve(Psi0,U)))
    UPsiU = suffStats[3]+np.dot(M0,la.solve(V0,M0.T))
    Psi = UPsi0U + np.dot(U.T, np.dot(UPsiU, U)) - np.dot(M, la.solve(V,M.T))

    return nu, Psi, M, V

def project_degenerate_transition_matrix(Fold, FU, U):
    """
    We can Gibbs sample a dimensionally compacted version of the transition
    matrix. This function projects it back out again.
    """

    ds = Fold.shape[0]
    F = np.dot( (np.identity(ds)-np.dot(U,U.T)), Fold ) + np.dot(U,FU)

    return F

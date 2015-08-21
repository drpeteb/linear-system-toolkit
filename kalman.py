import numpy as np
from scipy import linalg as la
from basic import GaussianDensity

def predict(dens, A, Q):
    """Kalman filter prediction"""
    prd_mn = np.dot(A,dens.mn)
    prd_vr = np.dot(A, np.dot(dens.vr,A.T) ) + Q
    prd_vr = (prd_vr+prd_vr.T)/2
    return GaussianDensity(prd_mn,prd_vr)

def correct(dens, y, H, R):
    """Kalman filter correction"""
    mu = np.dot(H,dens.mn)
    S = R + np.dot(H, np.dot(dens.vr,H.T) )
    S = (S+S.T)/2
    G = np.dot(dens.vr, la.solve(S,H,check_finite=False).T )
    I = np.identity(dens.dim)
    upd_vr = np.dot( I-np.dot(G,H), dens.vr)
    upd_vr = (upd_vr+upd_vr.T)/2
    upd_mn = dens.mn + np.dot(G, y-mu )
    return GaussianDensity(upd_mn,upd_vr), GaussianDensity(mu,S)
    
def update(flt, nxt_smt, nxt_prd, A):
    """Rauch-Tung-Striebel smoother update"""
    G = np.dot(flt.vr, la.solve(nxt_prd.vr,A,sym_pos=True,check_finite=False).T )
    smt_vr = flt.vr + np.dot(G, np.dot( nxt_smt.vr-nxt_prd.vr ,G.T) )
    smt_mn = flt.mn + np.dot(G, nxt_smt.mn-nxt_prd.mn )
    return GaussianDensity(smt_mn,smt_vr)
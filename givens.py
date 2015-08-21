# Givens matrix operations

import numpy as np
import scipy.linalg as la

def plane_rotation(x):
    """Find the plane rotation and Givens angle to zero a component"""
    if x[1] != 0:
        r = la.norm(x)
        G = np.hstack((np.expand_dims(x,1),np.expand_dims(x[::-1],1)))/r
        G[1,0] *= -1
        y = np.array([r,0])
    else:
        G = np.identity(2)
        y = x
    g = np.arctan(G[0,1]/G[0,0])
    return y,g,G
    
def givens_matrix(g,ii,jj,d):
    """create a givens rotation matrix"""
    G = np.identity(d)
    G[ii,ii] = np.cos(g)
    G[jj,jj] = np.cos(g)
    G[ii,jj] = np.sin(g)
    G[jj,ii] =-np.sin(g)
    return G

def givensise(U):
    """
    Factorise a matrix of orthonormal columns into row space and cross 
    rotations and a sign matrix, such that U = Ur x E x Uc
    """
    
    # Get the shape
    d,r = U.shape
    
    # Create arrays for the two rotation matrices
    Ur = np.identity(r)
    Uc = np.identity(d)
    
    # Row space loop
    for rr in reversed(range(r)):
        for cc in range(rr):
            
            # Get the next pair of elements to be rotated
            v = U[rr,[rr,cc]]
            
            # Find the plane rotation to zero the first element
            _,g,_ = plane_rotation(v)
            
            # Build the Givens matrix
            G = givens_matrix(g,cc,rr,r)

            # Perform the rotation 
            #TODO This is inefficient rotating the whole array - should do it on just the 4 elements affected.
            U = np.dot(U,G);
            Ur = np.dot(G.T,Ur)
            
    # Cross loop
    for cc in range(r):
        for rr in reversed(range(r,d)):
            
            # Get the next pair of elements to be rotated
            v = U[[cc,rr],cc]
            
            # Find the plane rotation to zero the first element
            _,g,_ = plane_rotation(v)
            
            # Build the Givens matrix
            G = givens_matrix(g,cc,rr,d)
            
            # Perform the rotation 
            #TODO This is inefficient rotating the whole array - should do it on just the 4 elements affected.
            U = np.dot(G,U)
            Uc = np.dot(Uc,G.T)
    
    # The remaining elements are the signs
    E = U
    
    return Uc,E,Ur
    
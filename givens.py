# Givens matrix operations

import numpy as np
import scipy.linalg as la
    
def plane_rotation(x):
    """Find the plane rotation and Givens angle to zero a component"""
    r = la.norm(x)
    x /= r
    if x[0] < 0:
        x *= -1
    if r > 0:
        G = np.hstack((x[:,np.newaxis],x[::-1,np.newaxis]))
        G[1,0] *= -1
    else:
        G = np.identity(2)
    
    #print(np.arctan2(x[1],x[0]))
    
    return G


def givensise(U):
    """
    Factorise a matrix of orthonormal columns into row space and cross 
    rotations and a sign matrix, such that U = Ur x E x Uc
    """
    
    # Copy it so we don't break it
    E = U.copy()
    
    # Get the shape
    d,r = E.shape
    
    # Create arrays for the two rotation matrices
    Ur = np.identity(r)
    Uc = np.identity(d)
    
    # Row space loop
    for rr in reversed(range(r)):
        for cc in range(rr):
            
            # Make a slice index for this row and column
            slce = slice(cc,rr+1,rr-cc)
            
            # Get the next pair of elements to be rotated
            v = E[rr,[rr,cc]]
            
            # Find the plane rotation to zero the first element
            G = plane_rotation(v)
            
            # Zero the element
            E[:,slce] = np.dot(E[:,slce],G)
            
            # Corresponding change in Ur
            Ur[slce,:] = np.dot(G.T,Ur[slce,:])
            
    # Cross loop
    for cc in range(r):
        for rr in reversed(range(r,d)):
            
            # Make a slice index for this row and column
            slce = slice(cc,rr+1,rr-cc)
            
            # Get the next pair of elements to be rotated
            v = E[[cc,rr],cc]
            
            # Find the plane rotation to zero the first element
            G = plane_rotation(v)
            
            # Zero the element in U
            E[slce,:] = np.dot(G,E[slce,:])
            
            # Corresponding change in Ur
            Uc[:,slce] = np.dot(Uc[:,slce],G.T)
            
    # The remaining elements are the signs
    E[np.isclose(E,0)] = 0
    
    return Uc,E,Ur
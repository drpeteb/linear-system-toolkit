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
    
def plane_rotation2(x):
    """Find the plane rotation and Givens angle to zero a component"""
    r = la.norm(x)
    if r > 0:
        G = np.hstack((x[:,np.newaxis],x[::-1,np.newaxis]))/r
        G[1,0] *= -1
    else:
        G = np.identity(2)
    return G
    
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
            
            print((rr,cc))
            
            # Get the next pair of elements to be rotated
            v = U[rr,[rr,cc]]
            
            # Find the plane rotation to zero the first element
            _,g,_ = plane_rotation(v)
            
            # Build the Givens matrix
            G = givens_matrix(g,cc,rr,r)

            # Perform the rotation 
            #TODO This is inefficient rotating the whole array - should do it on just the 4 elements affected.
            tmp = U.copy()
            U = np.dot(U,G);
            Ur = np.dot(G.T,Ur)
            
            #print(G)
            print(U)
            
    # Cross loop
    for cc in range(r):
        for rr in reversed(range(r,d)):
            
            print((rr,cc))
            
            # Get the next pair of elements to be rotated
            v = U[[cc,rr],cc]
            
            # Find the plane rotation to zero the first element
            _,g,_ = plane_rotation(v)
            
            # Build the Givens matrix
            G = givens_matrix(g,cc,rr,d)
            
            # Perform the rotation 
            #TODO This is inefficient rotating the whole array - should do it on just the 4 elements affected.
            tmp = U.copy()
            U = np.dot(G,U)
            Uc = np.dot(Uc,G.T)
            
            #print(G)
            print(U)
    
    # The remaining elements are the signs
    E = U
    
    return Uc,E,Ur


def givensise2(U):
    """
    Factorise a matrix of orthonormal columns into row space and cross 
    rotations and a sign matrix, such that U = Ur x E x Uc
    """
    
    # Copy it so we don't break it
    U = U.copy()
    
    # Get the shape
    d,r = U.shape
    
    # Create arrays for the two rotation matrices
    Ur = np.identity(r)
    Uc = np.identity(d)
    
    
    print("Row space")
    
    # Row space loop
    for rr in reversed(range(r)):
        for cc in range(rr):
            
            print((rr,cc))
            
            # Get the next pair of elements to be rotated
            v = U[rr,[rr,cc]]
            
            # Find the plane rotation to zero the first element
            G = plane_rotation2(v)
            
            # Zero the element in U
            U[:,cc:rr+1:rr-cc] = np.dot(U[:,cc:rr+1:rr-cc],G)
            
            # Corresponding change in Ur
            Ur[cc:rr+1:rr-cc,:] = np.dot(G.T,Ur[cc:rr+1:rr-cc,:])
            
            print(U)
    
    print("Cross")
    
    # Cross loop
    for cc in range(r):
        for rr in reversed(range(r,d)):
            
            print((rr,cc))
            
            # Get the next pair of elements to be rotated
            v = U[[cc,rr],cc]
            
            # Find the plane rotation to zero the first element
            G = plane_rotation2(v)
            
            # Zero the element in U
            U[cc:rr+1:rr-cc,:] = np.dot(G,U[cc:rr+1:rr-cc,:])
            
            # Corresponding change in Ur
            Uc[:,cc:rr+1:rr-cc] = np.dot(Uc[:,cc:rr+1:rr-cc],G.T)
            
            print(U)
    
    # The remaining elements are the signs
    E = U
    
    return Uc,E,Ur
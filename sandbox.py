import numpy as np
import scipy.linalg as la
from scipy import stats

# Givens matrix operations

import numpy as np
import scipy.linalg as la

def givens_matrix(d, r, c, g):
    assert(r!=c)
    if r > c:
        tmp = c
        c = r
        r = tmp
    U = np.identity(d)
    U[r,r] = np.cos(g)
    U[c,c] = np.cos(g)
    U[r,c] = -np.sin(g)
    U[c,r] = np.sin(g)
    return U

def plane_rotation(x):
    """Find the plane rotation and Givens angle to zero a component"""
    x = x.copy()
    r = la.norm(x)
    x /= r
    if x[0] < 0:
        x *= -1
    if r > 0:
        G = np.hstack((x[:,np.newaxis],x[::-1,np.newaxis]))
        G[1,0] *= -1
    else:
        G = np.identity(2)

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
        for rr in range(r,d):

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


d=5
r=3

Z = stats.norm.rvs(size=(d,r))
Q = np.dot(Z,Z.T)

val,vec = la.eigh(Q)
order = np.argsort(val)[::-1]
val = val[order]
vec = vec[:,order]
val = val[:r]
vec = vec[:,:r]

Uc,E,Ur = givensise(vec)

print(Uc)
print(E)
print(Ur)
print(np.allclose(np.dot(Uc,np.dot(E,Ur)), vec))

x = np.array([3.0,4.0])
g = plane_rotation(x)
print(g)
print(np.dot(g,x))

#A = np.arange(25)
#A.shape = (5,5)
#
#G = givens_matrix(5,2,4,0.15)
#
#print(A)
#print(G)
#print(np.dot(G,A))
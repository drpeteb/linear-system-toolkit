import numpy as np
import scipy.linalg as la

d = 4
r = 2

A = np.random.randn(d,d)
V = la.qr(A)[0][:,:r]

L = np.diag([2,1])
Q = np.dot(V, np.dot(L, V.T))

Uc = np.identity(d)
Ur = np.identity(r)
E = V.copy()

def givmat(d,i,j,g):
    G = np.identity(d)
    G[i,i] = np.cos(g)
    G[j,j] = np.cos(g)
    G[i,j] = -np.sin(g)
    G[j,i] = np.sin(g)
    return G

for i in reversed(range(r)):
    for j in range(i):

        angle = np.arctan(E[i,j]/E[i,i])
        G = givmat(r,i,j,angle)
        E = np.dot(E, G)
        Ur = np.dot(G.T, Ur)

        print(E)
        print('')

for j in range(r):
    for i in reversed(range(r,d)):

        angle = np.arctan(E[i,j]/E[j,j])
        G = givmat(d,i,j,angle)
        E = np.dot(G, E)
        Uc = np.dot(Uc, G.T)

        print(E)
        print('')

print(V)
print(np.dot(Uc, np.dot(E, Ur)))
print('')

U = Uc[:,:r]
Ed = E[:r,:r]
EUr = np.dot(Ed,Ur)
D = np.dot(EUr, np.dot(L, EUr.T))

Qr = np.dot(U, np.dot(D, U.T))

print(Q)
print(Qr)
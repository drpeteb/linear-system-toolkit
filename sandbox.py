import numpy as np
import scipy.linalg as la
from scipy import stats


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

print(np.allclose(Q,np.dot(np.dot(vec,np.diag(val)),vec.T)))


#z = stats.norm.rvs(size=(d,1))
#Qt = np.dot(z,z.T)
#new_vec = la.qr( np.hstack((vec,z)) )[0]
#valt = np.dot(new_vec.T,np.dot(Qt,new_vec))

print(vec)
print( la.eigh(np.dot(vec,vec.T))[0] )

S = np.dot(vec,vec.T)
Z = stats.multivariate_normal.rvs(mean=np.zeros(d),cov=S,allow_singular=True)

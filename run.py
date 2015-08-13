import numpy as np
from matplotlib import pyplot as plt
from linear_model import GaussianDensity, BasicLinearModel

plt.close('all')

K = 100
ds = 2
do = 1

params = dict()
params['F'] = np.array([[0.9,0.81],[0,0.9]])
params['Q'] = np.array([[1,0],[0,1]])
params['H'] = np.array([[1,0]])
params['R'] = np.array([[1]])

prior = GaussianDensity(np.array([0,0]), np.array([[100,0],[0,100]]))
model = BasicLinearModel(ds, do, prior, params)

state, observ = model.simulate_data(K)

fig = plt.figure()
for dd in range(ds):
    ax = fig.add_subplot(ds,1,dd+1)
    ax.plot(state[:,dd])

fig = plt.figure()
for dd in range(do):
    ax = fig.add_subplot(do,1,dd+1)
    ax.plot(observ[:,dd])

flt, prd, lhood = model.kalman_filter(observ)
smt = model.rts_smoother(flt, prd)

fig = plt.figure()
for dd in range(ds):
    ax = fig.add_subplot(ds,1,dd+1)
    ax.plot(flt.mn[:,dd], 'g-')
    ax.plot(flt.mn[:,dd]+2*np.sqrt(flt.vr[:,dd,dd]), 'g:')
    ax.plot(flt.mn[:,dd]-2*np.sqrt(flt.vr[:,dd,dd]), 'g:')

fig = plt.figure()
for dd in range(ds):
    ax = fig.add_subplot(ds,1,dd+1)
    ax.plot(smt.mn[:,dd], 'g-')
    ax.plot(smt.mn[:,dd]+2*np.sqrt(smt.vr[:,dd,dd]), 'g:')
    ax.plot(smt.mn[:,dd]-2*np.sqrt(smt.vr[:,dd,dd]), 'g:')
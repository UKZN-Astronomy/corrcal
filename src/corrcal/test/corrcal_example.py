import time

import numpy as np
from scipy.optimize import fmin_cg
from matplotlib import pyplot as plt

import corrcal
from corrcal import DATA_PATH


# scipy nonlinear conjugate gradient seems to work pretty well.
# note that it can use overly large step sizes in trials causing
# matrices to go non-positive definite.  If you rescale the gains
# by some large factor, this seems to go away.  If you routinely
# hit non-positive definite conditions, try increasing fac (or writing your
# own minimizer...)

# read relevant data.
f = open(DATA_PATH + '/ant1.dat')
ant1 = np.fromfile(f, 'int64') - 1
f.close()
f = open(DATA_PATH + '/ant2.dat')
ant2 = np.fromfile(f, 'int64') - 1
f.close()
f = open(DATA_PATH + '/gtmp.dat')
gvec = np.fromfile(f, 'float64')
f.close()
f = open(DATA_PATH + '/vis.dat')
data = np.fromfile(f, 'float64')
f.close()
mat = corrcal.read_sparse(DATA_PATH + '/signal_sparse2_test.dat')
# if you want to test timings, you can do so here.  Set t_min to some length of
# time, and code will see how many gradient operations it can get through
# during at least t_min seconds
# niter = 0
# t_min = 1e-4
# t1 = time.time()
# while (time.time() - t1) < t_min:
#     grad = corrcal2.get_gradient(gvec, data, mat, ant1, ant2)
#     niter = niter + 1
# t2 = time.time()
# nant = gvec.size / 2
# # time per gradient
# print( 'average time was ' + repr((time.time() - t1) / niter))
# # time (in microseconds) per visibility
# print( 'scaled_time was ' + repr((t2 - t1) / niter / nant / (nant - 1) * 1e6))

fac = 1000.0
t1 = time.time()

corrcal.get_chisq(gvec * fac, data, mat, ant1, ant2, scale_fac=fac)
corrcal.get_gradient(gvec * fac, data, mat, ant1, ant2, fac)
asdf = fmin_cg(corrcal.get_chisq, gvec * fac, corrcal.get_gradient,
               (data, mat, ant1, ant2, fac))
# t2 = time.time()
# print( 'elapsed time to do nonlinear fit for ' + repr(nant) + ' antennas
# was ' + repr(t2 - t1))
fit_gains = asdf / fac
plt.plot(data[1::2] ** 2 + data[::2] ** 2)
plt.show()

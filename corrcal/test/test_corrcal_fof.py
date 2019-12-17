import numpy as np
from scipy.optimize import fmin_cg

from corrcal import corrcal

nn = 5
x = np.arange(nn)
xmat = np.repeat([x], nn, axis=0)
ymat = xmat.copy().transpose()

scat = 0.01

xmat = xmat + scat * np.random.randn(xmat.shape[0], xmat.shape[1])
ymat = ymat + scat * np.random.randn(ymat.shape[0], ymat.shape[1])

xpos = np.reshape(xmat, xmat.size)
ypos = np.reshape(ymat, ymat.size)
antvec = np.arange(xpos.size)

xx = np.repeat([xpos], xmat.size, axis=0)
yy = np.repeat([ypos], ymat.size, axis=0).transpose()

antmat = np.repeat([antvec], antvec.size, axis=0)
ant1 = antmat.copy()
ant2 = antmat.copy().transpose()

umat = xx - xx.transpose()
vmat = yy - yy.transpose()
isok = np.where(ant2 > ant1)

ant1_org = ant1[isok]
ant2_org = ant2[isok]
u_org = umat[isok]
v_org = vmat[isok]
vis_org = np.random.randn(ant1_org.size) + \
    np.complex(0, 1) * np.random.randn(ant1_org.size)

noise_org = np.ones(u_org.size)

vis, u, v, noise, ant1, ant2, edges, ii, isconj = corrcal.grid_data(
    vis_org, u_org, v_org, noise_org, ant1_org, ant2_org
)
for i in range(len(edges) - 1):
    mystd = np.std(u[edges[i]:edges[i + 1]]) + np.std(
        v[edges[i]:edges[i + 1]])
    print((edges[i], edges[i + 1], mystd))
v1 = np.zeros(2 * vis.size)
v1[0::2] = 1
v2 = np.zeros(2 * vis.size)
v2[1::2] = 1
vecs = np.vstack([v1, v2])
src = v1 * 10

big_noise = np.zeros(2 * noise.size)
big_noise[0::2] = noise
big_noise[1::2] = noise

big_vis = np.zeros(2 * vis.size)
big_vis[0::2] = np.real(vis)
big_vis[1::2] = np.imag(vis)

mycov = corrcal.Sparse2Level(big_noise, 100 * vecs, 500 * src, 2 * edges)
guess = np.zeros(2 * len(ant1))
guess[0::2] = 1.0
fac = 1000.0

gvec = np.zeros(2 * ant1.max() + 2)
gvec[0::2] = 1.0
gvec = gvec + 0.1 * np.random.randn(gvec.size)
gvec[0] = 1
gvec[1] = 0

asdf = fmin_cg(corrcal.get_chisq, gvec * fac, corrcal.get_gradient,
               (big_vis + 500 * src, mycov, ant1, ant2, fac))

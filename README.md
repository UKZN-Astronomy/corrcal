# corrcal
`corrcal` is a calibration tool for radio interferometers that utilises the 
Correlation Calibration scheme developed in
[Sievers 2017](https://arxiv.org/abs/1701.01860)). This particular repository 
is being collaboratively developed to improve upon the original repository 
provided in Sievers 2017.


## Scientific Motivation
`corrcal` aims to bridge the gap between traditional sky based and redundancy 
based calibrations. Both extremes of calibrations rely on inherently incorrect 
assumptions:
* Sky based calibration relies on perfect sky models
* Redundant calibration relies on identical baselines

Both are, therefore, not realistic. _corrcal_ overcomes this by:
* relaxing the assumption of perfect redundancy
* enabling the inclusion of partial sky knowledge
    
It does so by describing visibility correlations, due to quasi-redundancy or 
foreground sources, with a covariance matrix C and reformulating the 
calibration problem as the following chi-square minimization:
<br/>
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<img width=200px src="https://render.githubusercontent.com/render/math?math=\chi^{2}=d^{\dagger}(N + GCG)^{-1}d">
<br/>

* d is the observed complex visibility (data) vector
* G is the complex gain matrix
* N is thermal noise matrix

This minimization requires efficient matrix inversion, which is achieved 
through Cholesky factorisation and the Woodbury identity. 

## Installation

Clone this repository,

```
git clone https://github.com/UKZN-Astronomy/corrcal.git
```

then `cd` into the cloned directory, and install through `pip`

```
pip install .
```


## Usage
Once code is set up, the important thing needed to run on ones data is
to get the sparse matrix describing sky and source correlations set
up. Please take a look at `read_sparse` in `corrcal.py` for an example of 
how the matrix is structured. A convenient function to convert `pyuvdata` 
object to `corrcal` sparse matrix is planned.
In short, the sparse matrix format groups visibilities into redundant blocks 
and separate the real and imaginary parts following their respective reals. 

The important fields in the spare matrix object are:
`diag`:  the noise variance of visibilities
`lims`:  the indices that set off the redundant blocks.
`vecs`:  the vectors describing the sky covariances within blocks.  It's
       currently assumed that the number of vectors is the same for
       each block.  If you don't want this, you can zero-pad.
`src`:   the per-visibility response to sources with known positions.
`isinv`: is the covariance matrix an inverse.  You will start with this 
       flag set to False.

When you have these in place, you can create a sparse matrix with
``` 
mat = corrcal.sparse_2level(diag, vecs, src, lims, isinv)
```

Note that if you want to run with classic redundant calibration, the
source vector will be zeros, and the sky vectors will be some large
number times

```
[1 0 1 0 1 0 1 0...
 0 1 0 1 0 1 0 1....]
```

which says there's random signal in the real visibilities which is
uncorrelated with the imaginary visibilities.  

To run, you will also need to get data and per-visibility antenna
1/antenna 2 (assumed zero-offset indexing on the antennas) read in,
plus a guess at the initial gains. Then you can
fit for gains with the scipy non-linear conjugate gradient solver
(`from scipy.optimize import fmin_cg`).  One final wrinkle is that `scipy`
often tries trial steps far too large for the gain errors, causing
matrix to go non-positive definite.  If you hit this, you can set a
scale factor to some large number until the minimizer behaves.  Look
in `corrcal_example.py` (which runs a whole PAPER-sized problem) to see
how this works.

## Contribution
`corrcal` is currently undergoing an active development and enhancement.
The current maintainers are:

* Piyanat Kittiwisit (University of KwaZulu-Natal)
* Ronniy Joseph (Curtin University)

with supervisation form Jonathan Sievers (University of KwaZulu-Natal and 
McGill University)

All contribution or suggestion are welcome. We are particularlly looking for
more codes developers.

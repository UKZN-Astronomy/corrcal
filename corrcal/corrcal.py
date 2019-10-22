import numpy
import ctypes
import time
import os


__all__ = ['sparse_2level', 'get_chisq', 'get_chisq_dense', 'get_gradient',
           'get_gradient_dense', 'chol', 'many_chol', 'tri_inv', 'many_tri_inv',
           'read_sparse', 'make_uv_grid', 'grid_data', 'mymatmul',
           'mult_vecs_by_blocks', 'make_uv_from_antpos']


try:
    import pyfof

    have_fof = True
except:
    have_fof = False


LOCATION = os.path.dirname(os.path.abspath(__file__))
# Change the following line to the corrcal dynamic library in you machine
mylib = ctypes.cdll.LoadLibrary(LOCATION + "/src/libcorrcal_funs.so")



sparse_matrix_vector_multiplication = mylib.sparse_mat_times_vec_wrapper
sparse_matrix_vector_multiplication.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_void_p]

make_small_block_c = mylib.make_small_block
make_small_block_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p]

make_all_small_blocks_c = mylib.make_all_small_blocks
make_all_small_blocks_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_void_p]
# void make_all_small_blocks(double *diag, double *vecs, long *lims, int nblock, int n, int nsrc, double *out)

cholesky_factorization = mylib.chol
cholesky_factorization.argtypes = [ctypes.c_void_p, ctypes.c_int]

cholesky_factorization_parallel = mylib.many_chol
cholesky_factorization_parallel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

tri_inv_c = mylib.tri_inv
tri_inv_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

many_tri_inv_c = mylib.many_tri_inv
many_tri_inv_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

invert_all_small_blocks_c = mylib.invert_all_small_blocks
invert_all_small_blocks_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

mymatmul_c = mylib.mymatmul
mymatmul_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

mult_vecs_by_blocs_c = mylib.mult_vecs_by_blocs
mult_vecs_by_blocs_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p]

apply_gains_to_matrix = mylib.apply_gains_to_mat
apply_gains_to_matrix.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_int]

apply_gains_to_mat_dense_c = mylib.apply_gains_to_mat_dense
apply_gains_to_mat_dense_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_int]

sum_grads_c = mylib.sum_grads
sum_grads_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


class sparse_2level:
    """
    sparse_2level contains/creates the data covariance matrix and all associated matrix operations

    Attributes
    ----------
    noise_variance  :   array_like
                        Contains the (1xN) thermal noise variance vector
    covariance_vectors  :   array_like
                            is an (MxN) array containing M eigenvectors of the data covariance matrix
    source_model_vectors    :   array_like
                                is an MxN array containing M model visibility vectors for M sources
    redudant_group_edges    :   array_like
                                is a vector that contains the start indices of each new redundant group as organised by
                                the grid_data function
    is_inverse  :   bool
                    ????? Probably something to use if you've have inverted the covariance matrices and did the
                    decomposition on that

    """


    def __init__(self, noise_variance, covariance_vectors, source_model_vectors, redundant_group_edges, is_inverse=0):
        self.noise_variance = noise_variance.copy()
        self.covariance_vectors = numpy.matrix(covariance_vectors.copy())
        self.source_model_vectors = numpy.matrix(source_model_vectors.copy())
        # self.lims=lims.copy()
        # We copy data like this because somethin something pointers that causes the C code to segault
        self.baseline_group_edges = numpy.zeros(len(redundant_group_edges), dtype='int64')
        self.baseline_group_edges[:] = redundant_group_edges
        self.is_inverse = is_inverse
        self.number_of_baseline_groups = len(redundant_group_edges) - 1

    def copy(self):
        """
        Method to copy sparse_2level object

        Returns:
            copy of sparse_2level object
        """
        return sparse_2level(self.noise_variance, self.covariance_vectors, self.source_model_vectors,
                             self.baseline_group_edges, self.is_inverse)

    #We define multiplication for this sparse2_level object
    def __mul__(self, input_vector):
        """
        Implements sparse matrix multiplication with any data length vector

        """

        result = numpy.zeros(input_vector.shape)
        # void sparse_mat_times_vec_wrapper(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, double *vec, double *ans)
        number_of_measurements = self.noise_variance.size
        number_of_covariance_vectors = self.covariance_vectors.shape[0]
        number_of_source_vectors = self.source_model_vectors.shape[0]
        sparse_matrix_vector_multiplication(self.noise_variance.ctypes.data, self.covariance_vectors.ctypes.data,
                                            self.source_model_vectors.ctypes.data, number_of_measurements,
                                            number_of_covariance_vectors, number_of_source_vectors,
                                            self.number_of_baseline_groups, self.baseline_group_edges.ctypes.data, self.is_inverse,
                                            input_vector.ctypes.data, result.ctypes.data)
        return result

    def expand(self):
        """
        Method ??????

        """
        m1 = self.source_model_vectors.transpose() * self.source_model_vectors
        for i in range(0, self.number_of_baseline_groups):
            i1 = self.baseline_group_edges[i]
            i2 = self.baseline_group_edges[i + 1]
            tmp = self.covariance_vectors[:, i1:i2].transpose() * self.covariance_vectors[:, i1:i2]
            m1[i1:i2, i1:i2] += tmp
        mm = numpy.diag(self.noise_variance)
        if (self.is_inverse):
            return mm - m1
        else:
            return mm + m1

    def inv(self):
        """
        Method that efficiently computes the inverse of a sparse matrix using provided c-code

        """
        t1 = time.time();
        myinv = self.copy()
        myinv.is_inverse = ~self.is_inverse
        myinv.noise_variance = 1.0 / self.noise_variance
        number_covariance_vectors = self.covariance_vectors.shape[0]
        number_of_measurements = self.noise_variance.size
        tmp = numpy.zeros([self.number_of_baseline_groups, number_covariance_vectors, number_covariance_vectors])

        make_all_small_blocks_c(self.noise_variance.ctypes.data, self.covariance_vectors.ctypes.data,
                                self.baseline_group_edges.ctypes.data, self.number_of_baseline_groups,
                                number_of_measurements, number_covariance_vectors, tmp.ctypes.data)

        myeye = numpy.repeat([numpy.eye(number_covariance_vectors)], self.number_of_baseline_groups, axis=0);
        if self.is_inverse:
            tmp2 = myeye - tmp
        else:
            tmp2 = myeye + tmp
        cholesky_factorization_parallel(tmp2.ctypes.data, number_covariance_vectors, self.number_of_baseline_groups)
        tmp3 = many_tri_inv(tmp2)
        tmp4 = mult_vecs_by_blocks(self.covariance_vectors, tmp3, self.baseline_group_edges)

        for i in range(tmp4.shape[0]):
            tmp4[i, :] = tmp4[i, :] * myinv.noise_variance
        # return tmp4
        # invert_all_small_blocks_c(tmp.ctypes.data,self.nblock,nvec,numpy.int(self.isinv),tmp2.ctypes.data)
        t2 = time.time()
        # print 'took ' + repr(t2-t1) + ' seconds to do inverse.'

        # return tmp,tmp3,tmp4
        myinv.covariance_vectors[:] = tmp4

        nsrc = self.source_model_vectors.shape[0]
        tmp = 0 * self.source_model_vectors
        n = self.noise_variance.size
        number_covariance_vectors = self.covariance_vectors.shape[0]
        nblock = self.baseline_group_edges.size - 1
        dptr = myinv.noise_variance.ctypes.data
        sptr = myinv.source_model_vectors.ctypes.data
        vptr = myinv.covariance_vectors.ctypes.data
        # we can do the block multiply simply by sending in 0 for nsrc
        for i in range(nsrc):
            sparse_matrix_vector_multiplication(dptr, vptr, sptr, n, number_covariance_vectors, 0, nblock, self.baseline_group_edges.ctypes.data, myinv.is_inverse,
                                                self.source_model_vectors[i].ctypes.data, tmp[i].ctypes.data)

        small_mat = tmp * self.source_model_vectors.transpose()

        if self.is_inverse:
            small_mat = numpy.eye(nsrc) - small_mat
        else:
            small_mat = numpy.eye(nsrc) + small_mat


        small_mat = numpy.linalg.inv(numpy.linalg.cholesky(small_mat))
        myinv.source_model_vectors = small_mat * tmp

        return myinv

    def apply_gains_to_mat(self, g, ant1, ant2):
        """
        Method that multiplies gain vector with the sparse representation of the data covariance matix G^T C G, using
        antenna indices for each baseline in the data vector it determines which gain is multiplied with which
        covariance entry

        Parameters
        ----------
        g   :   array_like
                (1xN) array that contains the gains of each antenna matching the indices in ant1 and ant2
        ant1    :   array_like
                    (1xN) array that contains the indices of the first antenna in each baseline corresponding the gain
                    in g
        ant2    :   array_like
                    (1xN) array that contains the indices of the second antenna in each baseline corresponding the gain
                    in g
        Returns
        -------

        None
        """
        apply_gains_to_matrix(self.covariance_vectors.ctypes.data, g.ctypes.data, ant1.ctypes.data, ant2.ctypes.data,
                              self.covariance_vectors.shape[1] // 2, self.covariance_vectors.shape[0])
        apply_gains_to_matrix(self.source_model_vectors.ctypes.data, g.ctypes.data, ant1.ctypes.data, ant2.ctypes.data,
                              self.source_model_vectors.shape[1] // 2, self.source_model_vectors.shape[0])

        return



def get_chisq_dense(g, data, noise, sig, ant1, ant2, scale_fac=1.0, normfac=1.0):
    """
    Not used
    """
    g = g / scale_fac
    cov = sig.copy()
    n = sig.shape[0]
    assert (sig.shape[1] == n)

    apply_gains_to_matrix(cov.ctypes.data, g.ctypes.data, ant1.ctypes.data, ant2.ctypes.data, n / 2, n)

    cov = cov.transpose().copy()
    apply_gains_to_matrix(cov.ctypes.data, g.ctypes.data, ant1.ctypes.data, ant2.ctypes.data, n / 2, n)

    cov = cov.transpose().copy()
    cov = cov + noise
    cov = 0.5 * (cov + cov.transpose())
    cov_inv = numpy.linalg.inv(cov)
    rhs = numpy.dot(cov_inv, data)
    chisq = numpy.sum(data * numpy.asarray(rhs))
    nn = g.size / 2
    chisq = chisq + normfac * ((numpy.sum(g[1::2])) ** 2 + (numpy.sum(g[0::2]) - nn) ** 2)
    print(chisq, numpy.mean(g[0::2]), numpy.mean(g[1::2]))
    return chisq


def get_gradient_dense(gains, data, covariance_matrix, sig, ant1, ant2, scale_fac=1.0, normfac=1.0):
    """
    Not used
    """
    do_times = False
    gains = gains / scale_fac
    cov = sig.copy()
    n = sig.shape[0]
    apply_gains_to_matrix(cov.ctypes.data, gains.ctypes.data, ant1.ctypes.data, ant2.ctypes.data, n / 2, n)
    cov = cov.transpose().copy()
    apply_gains_to_matrix(cov.ctypes.data, gains.ctypes.data, ant1.ctypes.data, ant2.ctypes.data, n / 2, n)
    cov = cov + covariance_matrix
    cov = 0.5 * (cov + cov.transpose())
    cov_inv = numpy.linalg.inv(cov)
    sd = numpy.dot(cov_inv, data)

    # make g*(c_inv)*d
    gsd = sd.copy()
    apply_gains_to_matrix(gsd.ctypes.data, gains.ctypes.data, ant2.ctypes.data, ant1.ctypes.data, n / 2, 1)

    cgsd = numpy.dot(gsd, sig)

    tmp = cgsd.copy()
    cgsd = numpy.zeros(tmp.size)
    cgsd[:] = tmp[:]

    tmp = sd.copy()
    sd = numpy.zeros(tmp.size)
    sd[:] = tmp[:]

    tmp = gsd.copy()
    gsd = numpy.zeros(tmp.size)
    gsd[:] = tmp[:]

    nant = numpy.max([numpy.max(ant1), numpy.max(ant2)]) + 1
    grad = numpy.zeros(2 * nant)

    r1 = gains[2 * ant1]
    r2 = gains[2 * ant2]
    i1 = gains[2 * ant1 + 1]
    i2 = gains[2 * ant2 + 1]
    m1r_v2 = 0 * cgsd
    m1i_v2 = 0 * cgsd
    m2r_v2 = 0 * cgsd
    m2i_v2 = 0 * cgsd

    m1r_v2[0::2] = r2 * sd[0::2] - i2 * sd[1::2];
    m1r_v2[1::2] = i2 * sd[0::2] + r2 * sd[1::2];
    m1i_v2[0::2] = i2 * sd[0::2] + r2 * sd[1::2];
    m1i_v2[1::2] = -r2 * sd[0::2] + i2 * sd[1::2];
    m2r_v2[0::2] = r1 * sd[0::2] + i1 * sd[1::2];
    m2r_v2[1::2] = -i1 * sd[0::2] + r1 * sd[1::2];
    m2i_v2[0::2] = i1 * sd[0::2] - r1 * sd[1::2];
    m2i_v2[1::2] = r1 * sd[0::2] + i1 * sd[1::2];

    if do_times:
        t2 = time.time();
        print(t2 - t1)

    v1_m1r_v2 = cgsd * m1r_v2;
    v1_m1r_v2 = v1_m1r_v2[0::2] + v1_m1r_v2[1::2];
    v1_m1i_v2 = cgsd * m1i_v2;
    v1_m1i_v2 = v1_m1i_v2[0::2] + v1_m1i_v2[1::2];
    v1_m2r_v2 = cgsd * m2r_v2;
    v1_m2r_v2 = v1_m2r_v2[0::2] + v1_m2r_v2[1::2];
    v1_m2i_v2 = cgsd * m2i_v2;
    v1_m2i_v2 = v1_m2i_v2[0::2] + v1_m2i_v2[1::2];
    if do_times:
        t2 = time.time();
        print(t2 - t1)

    # print v1_m1r_v2[0:5]

    sum_grads_c(grad.ctypes.data, v1_m1r_v2.ctypes.data, v1_m1i_v2.ctypes.data, ant1.ctypes.data, v1_m2i_v2.size)
    sum_grads_c(grad.ctypes.data, v1_m2r_v2.ctypes.data, v1_m2i_v2.ctypes.data, ant2.ctypes.data, v1_m2i_v2.size)
    if do_times:
        t2 = time.time();
        print(t2 - t1)
    # chisq=numpy.sum(sd*data)
    # print chisq

    nn = gains.size / 2.0
    grad_real = 2 * (numpy.sum(gains[0::2]) - nn) / nn
    grad_im = 2 * numpy.sum(gains[1::2])

    return -2 * grad / scale_fac + normfac * (grad_real + grad_im) / scale_fac


def get_chisq(gains, data, mat, ant1, ant2, scale_fac=1.0, normfac=1.0):
    """
    Calculates the the chi square for the current gains, data and data covariance matrix

    Args:
        gains   :   array_like
                    vector that contains the gain of each antenna
        data    :   array_like
                    vector that contains the visibility data split up into alternating real and imaginary pairs
        mat     :   sparse_2level object
                    sparse representation of the data covariance matrix
        ant1    :   array_like
                    vector that contains the antenna index of the first antenna in baseline, used to reference the gain
                    vector
        ant2    :   array_like
                    vector that contains the antenna index of the second antenna in baseline, used to reference the gain
                    vector
        scale_fac   :   float
                        arbritary scale factor that rescales the gains to help the minimiser actually reach a solution
                        (still unclear to me how this works)

        normfac :   float
                    unknown function

    Returns
    -------
    """
    gains = gains / scale_fac
    do_times = False
    if do_times:
        t1 = time.time()
    mycov = mat.copy()

    # Something is going wrong here due to numpy.array types
    mycov.apply_gains_to_mat(gains, ant1, ant2)

    if do_times:
        t2 = time.time();
        print(t2 - t1)
    mycov_inv = mycov.inv()
    if do_times:
        t2 = time.time();
        print(t2 - t1)
    sd = mycov_inv * data
    chisq = numpy.sum(sd * data)


    nn = gains.size / 2
    chisq = chisq + normfac * ((numpy.sum(gains[1::2])) ** 2 + (numpy.sum(gains[0::2]) - nn) ** 2)

    #print(chisq)
    #print(chisq)
    return chisq


def get_gradient(gains, data, covariance_matrix, antenna1_indices, antenna2_indices, scale_factor=1.0, normfac=1.0,
                 do_code_timing = False):
    """

    Parameters
    ----------
    gains   :   array_like
                A (1xN) vector containing the current gain solutions
    data    :   array_like
                A (1x2M) vector that contains the real and imaginary components of M measured visibilities.
    covariance_matrix   : sparse_2level object that contains the data covariance

    antenna1_indices    :   array_like
                            (1xM) vector that contains indices of the first antenna in a baseline pertaining referring to
                            the entries in gains
    antenna2_indices    :   array_like
                            (1xM) vector that contains indices of the second antenna in a baseline pertaining referring to
                            the entries in gains
    scale_factor    :   float
                        A scale factor that makes it easier for the scipy solver to converge.
    normfac :   float
                ???????? Unknown function
    do_code_timing  :   bool
                        Flag when timining output is desired

    Returns
        Gradient values at current evaluation of the chi-square
    -------

    """
    gains = gains / scale_factor

    if do_code_timing:
        t1 = time.time()

    covariance_matrix_copy = covariance_matrix.copy()
    covariance_matrix_copy.apply_gains_to_mat(gains, antenna1_indices, antenna2_indices)

    if do_code_timing:
        t2 = time.time();
        print(t2 - t1)

    covariance_matrix_inverse = covariance_matrix_copy.inv()

    if do_code_timing:
        t2 = time.time();
        print(t2 - t1)

    sd = covariance_matrix_inverse * data
    gsd = sd.copy();
    apply_gains_to_matrix(gsd.ctypes.data, gains.ctypes.data, antenna2_indices.ctypes.data, antenna1_indices.ctypes.data, gsd.size // 2, 1);
    tmp = covariance_matrix.copy()
    tmp.noise_variance[:] = 0
    cgsd = tmp * gsd

    if do_code_timing:
        t2 = time.time();
        print(t2 - t1)

    nant = numpy.max([numpy.max(antenna1_indices), numpy.max(antenna2_indices)]) + 1
    grad = numpy.zeros(2 * nant)

    r1 = gains[2 * antenna1_indices]
    r2 = gains[2 * antenna2_indices]
    i1 = gains[2 * antenna1_indices + 1]
    i2 = gains[2 * antenna2_indices + 1]
    m1r_v2 = 0 * cgsd
    m1i_v2 = 0 * cgsd
    m2r_v2 = 0 * cgsd
    m2i_v2 = 0 * cgsd

    m1r_v2[0::2] = r2 * sd[0::2] - i2 * sd[1::2];
    m1r_v2[1::2] = i2 * sd[0::2] + r2 * sd[1::2];
    m1i_v2[0::2] = i2 * sd[0::2] + r2 * sd[1::2];
    m1i_v2[1::2] = -r2 * sd[0::2] + i2 * sd[1::2];
    m2r_v2[0::2] = r1 * sd[0::2] + i1 * sd[1::2];
    m2r_v2[1::2] = -i1 * sd[0::2] + r1 * sd[1::2];
    m2i_v2[0::2] = i1 * sd[0::2] - r1 * sd[1::2];
    m2i_v2[1::2] = r1 * sd[0::2] + i1 * sd[1::2];

    if do_code_timing:
        t2 = time.time();
        print(t2 - t1)

    v1_m1r_v2 = cgsd * m1r_v2;
    v1_m1r_v2 = v1_m1r_v2[0::2] + v1_m1r_v2[1::2];
    v1_m1i_v2 = cgsd * m1i_v2;
    v1_m1i_v2 = v1_m1i_v2[0::2] + v1_m1i_v2[1::2];
    v1_m2r_v2 = cgsd * m2r_v2;
    v1_m2r_v2 = v1_m2r_v2[0::2] + v1_m2r_v2[1::2];
    v1_m2i_v2 = cgsd * m2i_v2;
    v1_m2i_v2 = v1_m2i_v2[0::2] + v1_m2i_v2[1::2];
    if do_code_timing:
        t2 = time.time();
        print(t2 - t1)

    # print v1_m1r_v2[0:5]

    sum_grads_c(grad.ctypes.data, v1_m1r_v2.ctypes.data, v1_m1i_v2.ctypes.data, antenna1_indices.ctypes.data, v1_m2i_v2.size)
    sum_grads_c(grad.ctypes.data, v1_m2r_v2.ctypes.data, v1_m2i_v2.ctypes.data, antenna2_indices.ctypes.data, v1_m2i_v2.size)
    if do_code_timing:
        t2 = time.time();
        print(t2 - t1)
    # chisq=numpy.sum(sd*data)
    # print chisq

    nn = gains.size / 2.0
    grad_real = 2 * (numpy.sum(gains[0::2]) - nn) / nn
    grad_im = 2 * numpy.sum(gains[1::2])
    return -2 * grad / scale_factor + normfac * (grad_real + grad_im) / scale_factor
    # return -2*grad/scale_fac


def chol(mat):
    """
    Wrapper function that passes sparse_2level matrix object to the c-code that performs cholesky factorisation

    Parameters
    ----------
    mat :   object
            sparse_2level matrix object that contains the data covariance

    Returns
    -------
        None
    """
    n = mat.shape[0]
    cholesky_factorization(mat.ctypes.data, n)


def many_chol(mat):
    """
    Wrapper function that passes the sparse_2level matrix object to the parallelised c-code that performs cholesky
     into a triangular matrix.

    Parameters
    ----------
    mat :   object
            sparse_2level matrix object that contains the data covariance

    Returns
    -------
        None
    """

    nmat = mat.shape[0]
    n = mat.shape[1]
    cholesky_factorization_parallel(mat.ctypes.data, n, nmat)


def tri_inv(mat):
    """
    Wrapper function that computes the inverse of the data covariance matrix, by passing the sparse_2level matrix object
    to the c-code that handles the inversion of a triangular matrix. Note: perform cholesky decomposition first?

    Parameters
    ----------
    mat :   object
            sparse_2level matrix object that contains the data covariance

    Returns
    -------
    mat_inv :   the inverse of the data covariance matrix
    """
    n = mat.shape[0]
    mat_inv = 0 * mat
    tri_inv_c(mat.ctypes.data, mat_inv.ctypes.data, n)
    return mat_inv


def many_tri_inv(mat):
    """
    Optimised matrix inversion?
    Wrapper function that computes the inverse of the data covariance matrix, by passing the sparse_2level matrix object
    to the c-code that handles the inversion of a triangular matrix. Note: perform cholesky decomposition first?

    Parameters
    ----------
    mat :   object
            sparse_2level matrix object that contains the data covariance

    Returns
    -------
    mat_inv :   the inverse of the data covariance matrix
    """

    mat_inv = 0 * mat
    sz = mat.shape

    # if only one matrix comes in, do the correct thing
    if len(sz) == 2:
        tri_inv_c(mat.ctypes.data, mat_inv.ctypes.data, sz[0])
        return mat_inv

    nmat = mat.shape[0]
    n = mat.shape[1]
    many_tri_inv_c(mat.ctypes.data, mat_inv.ctypes.data, n, nmat)
    return mat_inv


def read_sparse(fname):
    """
    Function that reads the covariance data set that comes with the corrcal2_example.py code
    Parameters
    ----------
    fname   :   str
                filename of the data file, should be binary

    Returns
    mat :   object
            sparse_2level object that contains all covariance information
    -------

    """

    f = open(fname)
    n = numpy.fromfile(f, 'int32', 1)[0];
    isinv = (numpy.fromfile(f, 'int32', 1)[0] != 0);
    nsrc = numpy.fromfile(f, 'int32', 1)[0];
    nblock = numpy.fromfile(f, 'int32', 1)[0];
    nvec = numpy.fromfile(f, 'int32', 1)[0];
    lims = numpy.fromfile(f, 'int32', (nblock + 1))
    diag = numpy.fromfile(f, 'float64', n)
    vecs = numpy.fromfile(f, 'float64', nvec * n)
    src = numpy.fromfile(f, 'float64', nsrc * n)
    crap = numpy.fromfile(f)
    f.close()



    if crap.size > 0:
        print('file ' + fname + ' had unexpected length.')
        return

    vecs = vecs.reshape([nvec, n])
    if nsrc > 0:
        src = src.reshape([nsrc, n])
    mat = sparse_2level(diag, vecs, src, lims, isinv)
    return mat


def make_uv_grid(u, v, tol=0.01, do_fof=True):
    """
    Function that grids data onto the uv-plane and to determine redundant groups

    Parameters
    ----------
    u   :   u-coordinates of each baseline (in wavelengths)
    v   :   v-coordinates of each baseline (in wavelengths)
    tol :   redundancy tolerance - withing what radius should baselines lie from each to be a redundant group. Depends
            on position precision of the array
    do_fof  :   bool
                if True it uses pyfof to find redundant groups. If False: it devides the uv_grid in block with size tol

    Returns
    -------
    ii  :   array_like
            Array that maps baselines as passed into make_uv_grid into redundant groups
    edges   :   array_like
                contains the starting index of each new redundant group

    iconj:  array_like
            Contains flags indicating whether visibilities are conjugated or not (e.g. which part of the uv-plane are
            they)
    """
    isconj = (v < 0) | ((v < tol) & (u < 0))
    u = u.copy()
    v = v.copy()
    u[isconj] = -1 * u[isconj]
    v[isconj] = -1 * v[isconj]
    if (have_fof & do_fof):
        uv = numpy.stack([u, v]).transpose()
        groups = pyfof.friends_of_friends(uv, tol)
        myind = numpy.zeros(len(u))
        for j, mygroup in enumerate(groups):
            myind[mygroup] = j
        ii = numpy.argsort(myind)
        edges = numpy.where(numpy.diff(myind[ii]) != 0)[0] + 1
    else:
        # break up uv plane into tol-sized blocks
        u_int = numpy.round(u / tol)
        v_int = numpy.round(v / tol)
        uv = u_int + numpy.complex(0, 1) * v_int
        ii = numpy.argsort(uv)
        uv_sort = uv[ii]
        edges = numpy.where(numpy.diff(uv_sort) != 0)[0] + 1
    edges = numpy.append(0, edges)
    edges = numpy.append(edges, len(u))

    # map isconj into post-sorting indexing
    isconj = isconj[ii]
    return ii, edges, isconj


def grid_data(vis, u, v, noise, ant1, ant2, tol=0.1, do_fof=True):
    """Re-order the data into redundant groups.  Inputs are (vis,u,v,noise,ant1,ant2,tol=0.1)
    where tol is the UV-space distance for points to be considered redundant.  Data will be
    reflected to have positive u, or positive v for u within tol of zero.  If pyfof is
    available, use that for group finding."""

    ii, edges, isconj = make_uv_grid(u, v, tol, do_fof)
    tmp = ant1[isconj]
    ant1[isconj] = ant2[isconj]
    ant2[isconj] = tmp
    vis = vis[ii]
    vis[isconj] = numpy.conj(vis[isconj])

    ant1 = ant1[ii]
    ant2 = ant2[ii]
    noise = noise[ii]

    return vis, u, v, noise, ant1, ant2, edges, ii, isconj


def mymatmul(a, b):
    n = a.shape[0]
    k = a.shape[1]
    kk = b.shape[0]
    m = b.shape[1]
    c = numpy.zeros([n, m])

    mymatmul_c(a.ctypes.data, k, b.ctypes.data, m, n, m, k, c.ctypes.data, m)
    return c


def mult_vecs_by_blocks(vecs, blocks, edges):
    n = vecs.shape[1]
    nvec = vecs.shape[0]
    nblock = edges.size - 1
    ans = numpy.zeros([nvec, n])
    if (edges.dtype.name != 'int64'):
        edges.numpy.asarray(edges, dtype='int64')
    mult_vecs_by_blocs_c(vecs.ctypes.data, blocks.ctypes.data, n, nvec, nblock, edges.ctypes.data, ans.ctypes.data)
    return ans


def make_uv_from_antpos(xyz, rmax=0, tol=0.0):
    """Take a list of antenna positions and create a UV snapshot out of it."""
    xyz = xyz.copy()
    nant = xyz.shape[0]
    if xyz.shape[1] == 2:
        xyz = numpy.c_[xyz, numpy.zeros(nant)]
    mymed = numpy.median(xyz, axis=0)
    xyz = xyz - numpy.repeat([mymed], nant, axis=0)

    if (rmax > 0):
        r = numpy.sqrt(numpy.sum(xyz ** 2, axis=1))
        xyz = xyz[r <= rmax, :]
        nant = xyz.shape[0]

    # fit to a plane by modelling z=a*x+b*y+c
    mat = numpy.c_[xyz[:, 0:2], numpy.ones(nant)]
    lhs = numpy.dot(mat.transpose(), mat)
    rhs = numpy.dot(mat.transpose(), xyz[:, 2])
    fitp = numpy.dot(numpy.linalg.inv(lhs), rhs)

    # now rotate into that plane.  z should now be vertical axis
    vz = numpy.dot(fitp, [1.0, 0, 0])
    vvec = numpy.asarray([1, 0, vz])
    vvec = vvec / numpy.linalg.norm(vvec)
    uz = numpy.dot(fitp, [0, 1.0, 0])
    uvec = numpy.asarray([0, 1, uz])
    uvec = uvec / numpy.linalg.norm(uvec)
    wvec = numpy.cross(uvec, vvec)
    rotmat = numpy.vstack([uvec, vvec, wvec]).transpose()
    xyz = numpy.dot(xyz, rotmat)

    x = xyz[:, 0].copy()
    xmat = numpy.repeat([x], nant, axis=0)
    y = xyz[:, 1].copy()
    ymat = numpy.repeat([y], nant, axis=0)
    antvec = numpy.arange(nant)

    ant1 = numpy.repeat([antvec], nant, axis=0)
    ant2 = ant1.copy().transpose()

    u = xmat - xmat.transpose()
    v = ymat - ymat.transpose()
    u = numpy.tril(u)
    v = numpy.tril(v)

    ii = (numpy.abs(u) > 0) & (numpy.abs(v) > 0)
    u = u[ii]
    v = v[ii]
    ant1 = ant1[ii]
    ant2 = ant2[ii]

    ii = (u < 0) | ((u < tol) & (v < 0))
    tmp = ant1[ii]
    ant1[ii] = ant2[ii]
    ant2[ii] = tmp
    u[ii] = u[ii] * -1
    v[ii] = v[ii] * -1

    return u, v, ant1, ant2, xyz

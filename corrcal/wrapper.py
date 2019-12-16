"""
wrapper.py uses `ctypes` to wrap functions in the corrcal C library
(c_corrcal*.so) to Python functions.

"""
import ctypes
import site
from glob import glob

__all__ = [
    'sparse_matrix_vector_multiplication', 'make_small_block_c',
    'make_all_small_blocks_c', 'cholesky_factorization',
    'cholesky_factorization_parallel', 'tri_inv_c', 'many_tri_inv_c',
    'invert_all_small_blocks_c', 'mymatmul_c', 'mult_vecs_by_blocs_c',
    'apply_gains_to_matrix', 'apply_gains_to_mat_dense_c', 'sum_grads_c'
]


# With ctypes, the location of libcorrcal*.so need to be hard-coded.
# This will not be needed once we migrate to CFFI.
LIB_LOC = glob(site.getsitepackages()[0] + "/c_corrcal*.so")[0]
mylib = ctypes.cdll.LoadLibrary(LIB_LOC)

sparse_matrix_vector_multiplication = mylib.sparse_mat_times_vec_wrapper
sparse_matrix_vector_multiplication.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p
]

make_small_block_c = mylib.make_small_block
make_small_block_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p]

make_all_small_blocks_c = mylib.make_all_small_blocks
make_all_small_blocks_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_void_p]
# void make_all_small_blocks(double *diag, double *vecs, long *lims,
# int nblock, int n, int nsrc, double *out)

cholesky_factorization = mylib.chol
cholesky_factorization.argtypes = [ctypes.c_void_p, ctypes.c_int]

cholesky_factorization_parallel = mylib.many_chol
cholesky_factorization_parallel.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_int]

tri_inv_c = mylib.tri_inv
tri_inv_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

many_tri_inv_c = mylib.many_tri_inv
many_tri_inv_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                           ctypes.c_int]

invert_all_small_blocks_c = mylib.invert_all_small_blocks
invert_all_small_blocks_c.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_void_p]

mymatmul_c = mylib.mymatmul
mymatmul_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

mult_vecs_by_blocs_c = mylib.mult_vecs_by_blocs
mult_vecs_by_blocs_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p]

apply_gains_to_matrix = mylib.apply_gains_to_mat
apply_gains_to_matrix.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int,
                                  ctypes.c_int]

apply_gains_to_mat_dense_c = mylib.apply_gains_to_mat_dense
apply_gains_to_mat_dense_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_int]

sum_grads_c = mylib.sum_grads
sum_grads_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_int]

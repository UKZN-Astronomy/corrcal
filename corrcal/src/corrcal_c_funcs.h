struct sparse_2level {
  double *diag;
  double *vecs;
  double *src;
  int n;
  int nvec;
  int nsrc;
  int nblock;
  long *lims;
  int isinv;
};

struct sparse_2level *fill_struct_sparse(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, int isinv);

//struct sparse_2level *mat=(struct sparse_2level *)malloc(sizeof(struct sparse_2level));
//
void chol(double *mat, int n);
//
void many_chol(double *mat, int n, int nmat);
//
void tri_inv(double *mat, double *mat_inv, int n);
//
void many_tri_inv(double *mat, double *mat_inv, int n, int nmat);
//
void mymatmul(double *a, int stridea, double *b, int strideb, int n, int m, int kk, double *c, int stridec);
//
void mult_vecs_by_blocs(double *vecs, double *blocks, int n, int nvec, int nblock, long *edges, double *ans);
//
//void apply_gains_to_mat_dense(double *mat, complex double *gains, long *ant1, long *ant2, int n, int nvec);
//
//void apply_gains_to_mat(complex double *mat, complex double *gains, long *ant1, long *ant2, int n, int nvec);
//
void sum_grads(double *grad, double *myr, double *myi, long *ant, int n);
//
void sparse_mat_times_vec(struct sparse_2level *mat, double *vec, double *ans);
//
void sparse_mat_times_vec_wrapper(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, int isinv, double *vec, double *ans);
//
void make_small_block(double *diag, double *vecs, int i1, int i2, int n, int nvec, double *out);
//
void make_all_small_blocks(double *diag, double *vecs, long *lims, int nblock, int n, int nvec, double *out);
//
void invert_all_small_blocks(double *blocks, int nblock, int nvec, int isinv, double *inv);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "corrcal_c_funcs.h"
#include <omp.h>
#include <math.h>
#include <complex.h>

//compile into a shared library with e.g.
//gcc-4.9 -fopenmp -std=c99 -O3 -shared -fPIC -o libcorrcal2_funs.so corrcal2_funs.c -lm -lgomp

struct sparse_2level *fill_struct_sparse(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, int isinv)
/*
 * Structure:  *fill_struct_sparse
 * --------------------
 * Fill structure of type sparse_2level with data. Allocate dynamic memory.
 *
 *  *diag: pointer to
 *  *vecs: pointer to
 *  *src: pointer to
 *  n:
 *  nvec:
 *  nsrc:
 *  nblock:
 *  *lims: pointer to
 *  isinv:
 *
 *  returns: pointer sparse_2level data structure
 */
 {
  struct sparse_2level *mat=(struct sparse_2level *)malloc(sizeof(struct sparse_2level));
  mat->diag=diag;
  mat->vecs=vecs;
  mat->src=src;
  mat->n=n;
  mat->nvec=nvec;
  mat->nsrc=nsrc;
  mat->nblock=nblock;
  mat->lims=lims;
  mat->isinv=isinv;


  return mat;
}
/*--------------------------------------------------------------------------------*/
//n=length(x);for j=1:n, l(j,j)=sqrt(x(j,j)-sum(l(j,1:j-1).^2));for k=j+1:n, l(k,j)=(x(j,k)-sum(l(j,1:j-1).*l(k,1:j-1)))/l(j,j);end;end;
void chol(double *mat, int n)
/*
 * Method:  chol(double *mat, int n)
 * --------------------
 * Performs Cholesky factorization on a matrix
 *
 *  *mat: pointer to matrix
 *  n: number of (vectors/columns?)
 *
 *  returns: none. result overwrites input, sored at *mat
 */
{
  for (int j=0;j<n;j++) {
    double tmp=mat[j*n+j];
    for (int i=0;i<j;i++)
      tmp -= mat[j*n+i]*mat[j*n+i];
    mat[j*n+j]=sqrt(tmp);
    double fac=1/mat[j*n+j];
    for (int k=j+1;k<n;k++) {
      tmp=mat[j*n+k];
      for (int i=0;i<j;i++)
	tmp -= mat[j*n+i]*mat[k*n+i];
      mat[k*n+j]=tmp*fac;
    }
  }

}
/*--------------------------------------------------------------------------------*/
void many_chol(double *mat, int n, int nmat)
/*
 * Method:  many_chol(double *mat, int n, int nmat)
 * --------------------
 * Performs Cholesky factorization on a list of matrices.
 * Supports OpenMP parallelisation.
 *
 *  *mat: pointer to matrices
 *  n: number of (vectors/columns?) per matrix
 *  nmat: number of matrices
 *
 *  returns: none. result overwrites input, stored at *mat
 */
{
#pragma omp parallel for
  for (int i=0;i<nmat;i++)
    chol(mat+i*n*n,n);
}

/*--------------------------------------------------------------------------------*/
void tri_inv(double *mat, double *mat_inv, int n)
/*
 * Method:  tri_inv(double *mat, double *mat_inv, int n)
 * --------------------
 * Inverts a lower-triangular matrix by (forward-substitution?)
 *
 *  *mat: pointer to matrix
 *  *mat_inv: pointer to inverse matrix
 *  n: number of (vectors/columns?) per matrix
 *
 *  returns: none. result stored at *mat_inv
 */
{
  for (int targ=0;targ<n;targ++) {
    mat_inv[targ*n+targ]=1/mat[targ*n+targ];
    for (int j=targ-1;j>=0;j--) {
      double tmp=0;
      for (int k=j+1;k<=targ;k++) {
	tmp+=mat[k*n+j]*mat_inv[targ*n+k];
      }
      mat_inv[j+n*targ]=-tmp/mat[j*n+j];
    }
  }

}
/*--------------------------------------------------------------------------------*/
void many_tri_inv(double *mat, double *mat_inv, int n, int nmat)
/*
 * Method:  many_tri_inv(double *mat, double *mat_inv, int n, int nmat)
 * --------------------
 * Invert a list of lower-triangular matrices by (forward-substitution?).
 * Supports OpenMP parallelisation.
 *
 *  *mat: pointer to matrices
 *  *mat_inv: pointer to inverse matrices
 *  n: number of (vectors/columns?) per matrix
 *  nmat: number of matrices
 *
 *  returns: none. result stored at *mat_inv
 */
{
#pragma omp parallel for
  for (int i=0;i<nmat;i++)
    tri_inv(mat+i*n*n,mat_inv+i*n*n,n);
}
/*--------------------------------------------------------------------------------*/
void mymatmul(double *a, int stridea, double *b, int strideb, int n, int m, int kk, double *c, int stridec)
/*
 * Method:  mymatmul(double *a, int stridea, double *b, int strideb, int n, int m, int kk, double *c, int stridec)
 * --------------------
 * Multiply two matrices together using ... ?
 *
 *  *a: pointer to first matrix
 *  stridea: (stride?) of first matrix
 *  *b: pointer to second matrix
 *  strideb: (stride?) of second matrix
 *  n: number of
 *  m: number of
 *  kk: number of
 *  *c: pointer to resultant matrix (answer)
 *  stridec: (stride?) of third matrix
 *
 *  returns: none. result stored at *c
 *
 * Avoids BLAS overhead for small multiplies.
 * If kk ever gets more than a few, should swap in a BLAS call
 */
{
  //double t1=omp_get_wtime();
  for (int i=0;i<n;i++)
    for (int j=0;j<m;j++)
      for (int k=0;k<kk;k++)
	c[i*stridec+j]+=a[i*stridea+k]*b[k*strideb+j];
  //double t2=omp_get_wtime();
  //printf("took %12.4e seconds.\n",t2-t1);
}
/*--------------------------------------------------------------------------------*/
void mult_vecs_by_blocs(double *vecs, double *blocks, int n, int nvec, int nblock, long *edges, double *ans)
/*
 * Method:  mult_vecs_by_blocs(double *vecs, double *blocks, int n, int nvec, int nblock, long *edges, double *ans)
 * --------------------
 * Multiply (vectors?) by (redundant blocks?)
 *
 *  *vecs: pointer to
 *  *blocks: pointer to
 *  n: number of
 *  nvec: number of
 *  nblock: number of (redundant blocks?)
 *  *edges: pointer to
 *  *ans: pointer to result
 *
 *  returns: none. result stored at *ans
 */
{
  //write out multiply for blocks.  If number of vecs gets more than a few, should
  //replace this with a BLAS call.
  for (int i=0;i<nblock;i++) {
    mymatmul(blocks+nvec*nvec*i,nvec,vecs+edges[i],n,nvec,edges[i+1]-edges[i],nvec,ans+edges[i],n);

    //vecs+edges[i],n,blocks+nvec*nvec*i,nvec,edges[i+1]-edges[i],nvec,nvec,ans+edges[i],n);
  }
}

/*--------------------------------------------------------------------------------*/
void apply_gains_to_mat_dense(double *mat, complex double *gains, long *ant1, long *ant2, int n, int nvec)
/*
 * Method:  apply_gains_to_mat_dense(double *mat, complex double *gains, long *ant1, long *ant2, int n, int nvec)
 * --------------------
 * Apply gains to (dense matrix?) mat. Uses dynamic memory allocation.
 *
 *  *mat: pointer to
 *  *gains: pointer to
 *  *ant1: pointer to
 *  *ant2: pointer to
 *  n: number of
 *  nvec: number of
 *
 *  returns: none. result overwrites input, stored at *mat
 */
{
  complex double *gvec=(double complex *)malloc(sizeof(double complex)*n/2);
  for (int i=0;i<n/2;i++) {
    gvec[i]=(gains[ant1[i]])*conj(gains[ant2[i]]);
  }

  for (int j=0;j<nvec;j++){
    for (int i=0;i<n/2;i++) {
      complex double tmp=mat[j*n+2*i]+_Complex_I*mat[j*n+2*i+1];
      tmp=gvec[i]*tmp;
      mat[j*n+2*i]=creal(tmp);
      mat[j*n+2*i+1]=cimag(tmp);
    }
  }
  free(gvec);
}
/*--------------------------------------------------------------------------------*/
void apply_gains_to_mat(complex double *mat, complex double *gains, long *ant1, long *ant2, int n, int nvec)
/*
 * Method:  apply_gains_to_mat(complex double *mat, complex double *gains, long *ant1, long *ant2, int n, int nvec)
 * --------------------
 * Apply gains to (sparse matrix?) mat. Uses dynamic memory allocation.
 *
 *  *mat: pointer to
 *  *gains: pointer to
 *  *ant1: pointer to
 *  *ant2: pointer to
 *  n: number of
 *  nvec: number of
 *
 *  returns: none. result overwites input, stored at *mat
 */
{
  //printf("%d", creal(gains));
  double complex *gvec=(double complex *)malloc(sizeof(double complex)*n);
  //printf("n,nvec are %d %d\n",n,nvec);
  for (int i=0;i<n;i++) {
    gvec[i]=(gains[ant1[i]])*conj(gains[ant2[i]]);
    //gvec[i]=conj(gains[ant1[i]])*(gains[ant2[i]]);
  }
  //printf("made gvec.\n");
  for (int i=0;i<nvec;i++)
    for (int j=0;j<n;j++)
      mat[i*n+j]*=gvec[j];
  //printf("applied gains.\n");
  free(gvec);
}
/*--------------------------------------------------------------------------------*/
void sum_grads(double *grad, double *myr, double *myi, long *ant, int n)
/*
 * Method:  sum_grads(double *grad, double *myr, double *myi, long *ant, int n)
 * --------------------
 * Sum up (gradients?)
 *
 *  *grad: pointer to the result
 *  *myr: pointer to
 *  *myi: pointer to
 *  *ant: pointer to
 *  n: number of
 *
 *  returns: none. result stored at *grad
 */
{
  for (int i=0;i<n;i++) {
    grad[2*ant[i]]+=myr[i];
    grad[2*ant[i]+1]+=myi[i];
  }
}

/*--------------------------------------------------------------------------------*/
void sparse_mat_times_vec(struct sparse_2level *mat, double *vec, double *ans)
/*
 * Method:  sparse_mat_times_vec(struct sparse_2level *mat, double *vec, double *ans)
 * --------------------
 * Multiplies a sparse matrix by a vector.
 *
 *  *mat: pointer to sparse matrix (of type sparse_2level)
 *  *vec: pointer to vector
 *  *ans: pointer to result
 *
 *  returns: none. result stored at *ans
 */
{
  double t1=omp_get_wtime();
  memset(ans,0,sizeof(double)*mat->n);
  for (int i=0;i<mat->n;i++)
    ans[i]=vec[i]*mat->diag[i];

  if (mat->isinv) {
    for (int i=0;i<mat->nblock;i++) {
      for (int j=0;j<mat->nvec;j++) {
	double tot=0;
	for (int k=mat->lims[i];k<mat->lims[i+1];k++)
	  tot+=vec[k]*mat->vecs[j*mat->n+k];
	for (int k=mat->lims[i];k<mat->lims[i+1];k++)
	  ans[k]-=tot*mat->vecs[j*mat->n+k];

      }
    }
    for (int i=0;i<mat->nsrc;i++) {
      double tot=0;
      for (int j=0;j<mat->n;j++)
	tot+=vec[j]*mat->src[i*mat->n+j];
      for (int j=0;j<mat->n;j++)
	ans[j]-=tot*mat->src[i*mat->n+j];
    }
  }
  else {
    for (int i=0;i<mat->nblock;i++) {
      for (int j=0;j<mat->nvec;j++) {
	double tot=0;
	for (int k=mat->lims[i];k<mat->lims[i+1];k++)
	  tot+=vec[k]*mat->vecs[j*mat->n+k];
	for (int k=mat->lims[i];k<mat->lims[i+1];k++)
	  ans[k]+=tot*mat->vecs[j*mat->n+k];

      }
    }
    for (int i=0;i<mat->nsrc;i++) {
      double tot=0;
      for (int j=0;j<mat->n;j++)
	tot+=vec[j]*mat->src[i*mat->n+j];
      for (int j=0;j<mat->n;j++)
	ans[j]+=tot*mat->src[i*mat->n+j];
    }


  }
  double t2=omp_get_wtime();
  //printf("took %12.6f seconds to multiply.\n",t2-t1);
}

/*--------------------------------------------------------------------------------*/

void sparse_mat_times_vec_wrapper(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, int isinv, double *vec, double *ans)
/*
 * Method:  sparse_mat_times_vec_wrapper(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, long *lims, int isinv, double *vec, double *ans)
 * --------------------
 * Wrapper which multiplies a sparse matrix by a vector.
 * Fill sparse_2level data-structure, perform multiplication, free up allocated memory.
 *
 *  *diag: pointer to
 *  *vecs: pointer to
 *  *src: pointer to
 *  n:
 *  nvec:
 *  nsrc:
 *  nblock:
 *  *lims: pointer to
 *  isinv:
 *  *vec: pointer to vector
 *  *ans: pointer to output result
 *
 *  returns: none. result stored at *ans
 */
{
  struct sparse_2level *mat=fill_struct_sparse(diag,vecs,src,n,nvec,nsrc,nblock,lims,isinv);
  sparse_mat_times_vec(mat,vec,ans);
  free(mat);
}
//struct sparse_2level *mat, double *vec, double *ans)
//struct sparse_2level fill_struct_sparse(double *diag, double *vecs, double *src, int n, int nvec, int nsrc, int nblock, int *lims)
/*--------------------------------------------------------------------------------*/
void make_small_block(double *diag, double *vecs, int i1, int i2, int n, int nvec, double *out)
/*
 * Method:  make_small_block(double *diag, double *vecs, int i1, int i2, int n, int nvec, double *out)
 * --------------------
 * Makes a small block?
 *
 *  *diag: pointer to
 *  *vecs: pointer to
 *  i1:
 *  i2:
 *  n: number of
 *  nvec:
 *  *out: pointer to output result
 *
 *  returns: none. result stored at *out
 */
{

  for (int i=0;i<nvec;i++)
    for (int j=i;j<nvec;j++) {
      for (int k=i1;k<i2;k++)
	out[i*nvec+j]+=vecs[i*n+k]*vecs[j*n+k]/diag[k];
      out[j*nvec+i]=out[i*nvec+j];
    }
}
/*--------------------------------------------------------------------------------*/
void make_all_small_blocks(double *diag, double *vecs, long *lims, int nblock, int n, int nvec, double *out)
/*
 * Method:  make_all_small_blocks(double *diag, double *vecs, long *lims, int nblock, int n, int nvec, double *out)
 * --------------------
 * Makes many small blocks?
 *
 *  *diag: pointer to
 *  *vecs: pointer to
 *  *lims: pointer to
 *  nblock:
 *  n: number of
 *  nvec:
 *  *out: pointer to output result
 *
 *  returns: none. result stored at *out
 */
{
  for (int i=0;i<nblock;i++) {
    make_small_block(diag,vecs,lims[i],lims[i+1],n,nvec,out+i*nvec*nvec);
  }
}
/*--------------------------------------------------------------------------------*/
void invert_all_small_blocks(double *blocks, int nblock, int nvec, int isinv, double *inv)
/*
 * Method:  invert_all_small_blocks(double *blocks, int nblock, int nvec, int isinv, double *inv)
 * --------------------
 * Inverts all small blocks?
 *
 *  *blocks: pointer to
 *  nblock:
 *  nvec: number of
 *  isinv:
 *  *inv: pointer to output result
 *
 *  returns: none. result stored at *inv
 */
{

  for (int i=0;i<nblock;i++) {
    int istart=i*nvec*nvec;
    if (isinv)
      for (int j=0;j<nvec*nvec;j++)
	blocks[istart+j]*=-1;

    //add 1 to diagonal
    for (int j=0;j<nvec;j++) {
      int ii=istart+j*nvec+j;
      //printf("%d %d\n",ii,nblock*nvec*nvec);
      blocks[istart+j*nvec+j]++;
    }

    chol(blocks+istart,nvec);
    tri_inv(blocks+istart,inv+istart,nvec);

#if 1
    //be civilized and zero out upper triangle
    for (int j=0;j<nvec;j++)
      for (int k=j+1;k<nvec;k++)
	blocks[istart+j*nvec+k]=0;
#endif
  }
}
/*--------------------------------------------------------------------------------*/

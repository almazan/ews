#include <assert.h>
#include <string.h>

#include <omp.h>
#include <immintrin.h>

#include "x_matrix.h"


#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))
#define NEWAC(type,n) (type*)calloc(sizeof(type),(n))
#define NEW(type) NEWA(type,1)


// #define FASTCACHE

void vec_scale(float *w, long d, float fw) {

  if(((long)w & 15) == 0 && (d & 3) == 0 ) {        
    __v4sf *w4 = (void*)w; 
    __v4sf a4 = {fw, fw, fw, fw};
    d /= 4;

#ifdef FASTCACHE
    while(d--) *w4 *= a4; 
#else 
    while(d--) *w4++ *= a4; 
#endif

    return;
  } 

  
  /* reference version */
  int j;
  for(j = 0; j < d; j++) 
    w[j] *= fw;
}


/* squared norm of vector */
float vec_sqnorm(const float *w, long d) {

  
  if(((long)w & 15) == 0 && (d & 3) == 0 ) {        
    __v4sf *w4 = (void*)w; 
    __v4sf accu4 = {0, 0, 0, 0};
    d /= 4;
    
    while(d--) {
      accu4 += (*w4) * (*w4); 
#ifndef FASTCACHE
      w4++; 
#endif
    }
    
    float *accu = (void*)&accu4;         
    return accu[0] + accu[1] + accu[2] + accu[3];
  } 

  /* reference version */
  long j;
  double accu = 0;
  for(j = 0; j < d; j++) 
    accu += w[j] * w[j];
  return accu;

}


/* dot product */
float vec_dotprod(const float * __restrict__ xi, 
                         const float * __restrict__ w, long d) {
  
  if(((long)xi & 15) == 0 && ((long)w & 15) == 0 && (d & 3) == 0 ) {        
    __v4sf *xi4 = (void*)xi; 
    __v4sf *w4 = (void*)w; 
    __v4sf accu4 = {0, 0, 0, 0};
    d /= 4;
#ifndef FASTCACHE
    while(d--) accu4 += (*xi4++) * (*w4++); 
#else
    while(d--) accu4 += (*xi4) * (*w4); 
#endif
    
    float *accu = (void*)&accu4;         
    return accu[0] + accu[1] + accu[2] + accu[3];
  } 

  /* reference version */
  long j;
  double accu = 0;
  for(j = 0; j < d; j++) 
    accu += xi[j] * w[j];
  return accu;
}

/* w += xi * a */
void vec_addto(float * __restrict__ w,
                      float a, 
                      const float * __restrict__ xi, 
                      long d) {

  if(((long)xi & 15) == 0 && ((long)w & 15) == 0 && (d & 3) == 0 ) {        
    __v4sf *xi4 = (void*)xi; 
    __v4sf *w4 = (void*)w; 
    __v4sf a4 = {a, a, a, a};
    d /= 4;
#ifdef FASTCACHE
    while(d--) *w4 += (*xi4) * a4; 
#else
    while(d--) *w4 ++ += (*xi4++) * a4; 
#endif
    return;
  } 

  /* reference version */
  long j; 
  for(j = 0; j < d; j++) 
    w[j] += xi[j] * a;
  return;
  
}


/* return x(:, i) */
const float *x_matrix_get(const x_matrix_t *x,
                                 int i,
                                 float *buffer) { /* buffer of size d that may or may not be used */
  
  switch(x->encoding) {

  case JSGD_X_FULL:
    return x->data + x->d * i;

  case JSGD_X_PQ:
    {
      int q, nsq = x->nsq, ksq = x->d / x->nsq;       
      float *cent = x->centroids; 
      unsigned char *c = x->codes; 
      for(q = 0; q < nsq; q++) {
        memcpy(buffer + q * ksq, cent + ksq * c[q], sizeof(float) * ksq); 
        cent += ksq * 256; 
      }
    }
    return buffer; 
  case JSGD_X_SPARSE: 
    {
      memset(buffer, 0, sizeof(float) * x->d); 
      long nz;
      for(nz = x->indptr[i]; nz < x->indptr[i + 1]; nz++) 
        buffer[x->indices[nz]] = x->sparse_data[nz];
    }
    return buffer;
  default: 
    assert(!"not implemented");
  }

}

/* return w * x(:, i) */
double x_matrix_dotprod_d_slice(const x_matrix_t *x,
                               long i,
                                const float *w, 
                                int d0, int dd) {

  
  switch(x->encoding) {

  case JSGD_X_FULL:
    return vec_dotprod(x->data + x->d * i + d0, w, dd); 

  case JSGD_X_PQ: 
    {
      int q, nsq = x->nsq, ksq = x->d / nsq;       /* rename ksq as ds */
      int q0 = d0 / ksq, q1 = q0 + dd / ksq;
      const float *cent = x->centroids + q0 * 256 * ksq; 
      const unsigned char *c = x->codes + i * nsq; 
      double accu = 0; 
      for(q = q0; q < q1; q++) {
        accu += vec_dotprod(w, cent + ksq * c[q], ksq); 
        w += ksq;
        cent += ksq * 256; 
      }
      return accu; 
    }    
  case JSGD_X_SPARSE: 
    {
      assert(d0 == 0 && dd == x->d); /* else not implemented */
      long nz;
      double accu = 0; 
      for(nz = x->indptr[i]; nz < x->indptr[i + 1]; nz++) 
        accu += w[x->indices[nz]] * x->sparse_data[nz];
      return accu;
    }
  default: 
    assert(!"not implemented");
  }

}

double x_matrix_dotprod(const x_matrix_t *x,
                        long i,
                        const float *w) {
  return x_matrix_dotprod_d_slice(x, i, w, 0, x->d);  
}

                                 
/* w := w + a * x(:, i)  */
static void x_matrix_addto_d_slice(const x_matrix_t *x,
                            long i,
                            float a,
                            float *w, 
                            int d0, int dd) { 
  
  switch(x->encoding) {

  case JSGD_X_FULL:
    vec_addto(w, a, x->data + x->d * i + d0, dd); 
    return; 

  case JSGD_X_PQ: 
    {
      int q, nsq = x->nsq, ksq = x->d / nsq;       
      int q0 = d0 / ksq, q1 = q0 + dd / ksq;
      const float *cent = x->centroids + ksq * 256 * q0; 
      const unsigned char *c = x->codes + i * nsq; 
      for(q = q0; q < q1; q++) {
        vec_addto(w, a, cent + ksq * c[q], ksq); 
        w += ksq;
        cent += ksq * 256; 
      }
      return; 
    }    
  case JSGD_X_SPARSE: 
    {
      assert(d0 == 0 && dd == x->d); /* else not implemented */
      long nz;
      for(nz = x->indptr[i]; nz < x->indptr[i + 1]; nz++) 
        w[x->indices[nz]] += a * x->sparse_data[nz];
      return;
    }    
  default: 
    assert(!"not implemented");
  }
}

void x_matrix_addto(const x_matrix_t *x,
                            long i,
                            float a,
                            float *w) {
  x_matrix_addto_d_slice(x, i, a, w, 0, x->d);
}


double x_matrix_dotprod_self(const x_matrix_t *x,
                             long i, long j) {

  switch(x->encoding) {

  case JSGD_X_FULL:
    return vec_dotprod(x->data + x->d * i, x->data + x->d * j, x->d); 

  case JSGD_X_PQ: 
    {
      int q, nsq = x->nsq, ksq = x->d / nsq;       
      const float *cent = x->centroids; 
      const unsigned char *ci = x->codes + i * nsq; 
      const unsigned char *cj = x->codes + j * nsq; 
      double accu = 0; 
      for(q = 0; q < nsq; q++) {
        accu += vec_dotprod(cent + ksq * ci[q], cent + ksq * cj[q], ksq); 
        cent += ksq * 256; 
      }
      return accu; 
    }       

  case JSGD_X_SPARSE: 
    {
      long nzi = x->indptr[i], nzi1 = x->indptr[i + 1];
      long nzj = x->indptr[j], nzj1 = x->indptr[j + 1];
      double accu = 0; 
      while(nzi < nzi1 && nzj < nzj1) { 
        long ii = x->indices[nzi], ij = x->indices[nzj];
        if(ii == ij) {
          accu += x->sparse_data[nzi] * x->sparse_data[nzj];
          nzi ++; nzj++; 
        } else if(ii < ij) nzi++; 
        else nzj++; 
      }
      return accu;
    }    
  default: 
    assert(!"not implemented");
  }

}
                                 

#define real float

/* the integer type should be provided via 
 * 
 *  -Dinteger=int or -Dinteger=long 
 *
 * on the compiler command line. This depends on the used Blas
 * implementation */

int sgemm_ (char *transa, char *transb, integer * m, integer *
            n, integer * k, real * alpha, const real * a, integer * lda,
            const real * b, integer * ldb, real * beta, real * c__,
            integer * ldc);


int ssyrk_(char *uplo, char *trans, integer *n, integer *k, 
           real *alpha, real *a, integer *lda, real *beta, real *c__, integer *
           ldc);


void x_matrix_matmul_slice(const x_matrix_t *x,
                           int i0, int i1,
                           const float *w,
                           int m, 
                           float *scores) {
  const long d = x->d; 

  if(x->encoding == JSGD_X_FULL) { /* use blas3 */
    integer mi = m, di = d, ni = i1 - i0; 
    float one = 1.0, zero = 0.0;    
    sgemm_("Transposed", "Not", &mi, &ni, &di, &one, w, &di, 
           x->data + i0 * x->d, &di, &zero, scores, &mi);     
    return; 
  }

  int i, j;

  for(i = i0; i < i1; i++) {
    for(j = 0; j < m; j++) 
      scores[(i - i0) * m + j ] = x_matrix_dotprod(x, i, w + j * d);
  }
}

/* sizes:
 *
 * x(d, n) 
 * w(d, m)
 * scores(m, n) 
 *
 * scores = w' * x */

void x_matrix_matmul(const x_matrix_t *x,
                            const float *w, 
                            int m, 
                            float *scores) {
  x_matrix_matmul_slice(x, 0, x->n, w, m, scores);
}


void x_matrix_matmul_thread(const x_matrix_t *x,
                                   const float *w, 
                                   int m, 
                                   float *scores) {

  /* Matlab's Blas does not like to be called from OpenMP */
#ifdef BLAS_WITH_THREADS
  if(x->encoding == JSGD_X_FULL) {
    x_matrix_matmul(x, w, m, scores);
    return; 
  }
#endif

  /* split over n */
  int nt = omp_get_max_threads(); 
  
  int i; 
#pragma omp parallel for
  for(i = 0; i < nt; i++) {    
    int i0 = i * x->n / nt, i1 = (i + 1) * x->n / nt;     
    x_matrix_matmul_slice(x, i0, i1, w, m, scores + m * i0);
  }
}



void x_matrix_sparse_init(x_matrix_sparse_t *a, int m, int n, int nnz) {
  char * buffer = malloc(sizeof(float) * nnz + 
                         sizeof(int) * (n + 1 + nnz));
  a->pr = (void*)buffer; buffer += sizeof(float) * nnz; 
  a->jc = (void*)buffer; buffer += sizeof(int) * (n + 1); 
  a->ir = (void*)buffer; 
  a->m = m; 
  a->n = n; 
}

void x_matrix_sparse_clear(x_matrix_sparse_t *a) {
  free(a->pr); 
  a->m = a->n = 0; 
  a->pr = NULL; 
  a->jc = a->ir = NULL; 
}



void x_matrix_matmul_self(const x_matrix_t *x,
                          int i0, int i1, 
                          float *y) {
#if 0
  if(x->encoding == JSGD_X_FULL) { /* use blas3 */
    integer mi = m, di = d, ni = i1 - i0; 
    float one = 1.0, zero = 0.0;    
    sgemm_("Transposed", "Not", &mi, &ni, &di, &one, w, &di, 
           x->data + i0 * x->d, &di, &zero, scores, &mi);     
    return; 
  }
#endif

  int i, j; 
  
  i1 -= i0;

  for(i = 0; i < i1; i++) 
    for(j = i; j < i1; j++) 
      y[i + j * i1] = 
        y[j + i * i1] = 
        x_matrix_dotprod_self(x, i + i0, j + i0); 
}



#define D_STEP 1024

void x_matrix_matmul_subset(const x_matrix_t *x,
                            x_matrix_sparse_t *a, 
                            const float *ws, 
                            int nw, 
                            int d_step) {

  const int *is = a->ir, *is_begins = a->jc; 
  float *y = a->pr;
  int i, j;   

  if(is_begins[0] == is_begins[nw]) return;
#if 0
  /* does not seem to improve on 134 classes * 50 examples in 4096D with 100/134 density ???*/
  if(x->encoding == JSGD_X_FULL) {
    /* then it may be less expensive to compute all values with BLAS */
    int i0 = 10000000, i1 = -i0; 
    /* find bounds in i dimension */
    for(j = 0; j < nw; j++) if(is_begins[j + 1] > is_begins[j]) {
      int ii0 = is[is_begins[j]]; 
      if(ii0 < i0) i0 = ii0;
      int ii1 = is[is_begins[j + 1] - 1]; 
      if(ii1 > i1) i1 = ii1;
    }
    i1 ++; 
    int nnz = is_begins[nw] - is_begins[0];
    if(1 /* nnz > nw * (i1 - i0) / 5 */) { /* more than 1/5th of the values are needed: compute all */
      float *full_matrix = NEWA(float, nw * (i1 - i0));
     
      x_matrix_matmul_slice(x, i0, i1, ws, nw, full_matrix); 

      for(i = 0; i < nw; i++) 
        for(j = is_begins[i]; j < is_begins[i + 1]; j++) 
          y[j] = full_matrix[i + (is[j] - i0) * nw];      
      free(full_matrix);
      return;
    }
  }
#endif


  if(x->d % d_step != 0) {
    for(i = 0; i < nw; i++) {
      const float *w = ws + i * x->d; 
      for(j = is_begins[i]; j < is_begins[i + 1]; j++) 
        y[j] = x_matrix_dotprod(x, is[j], w); 
    }
  } else {
    int d;
    memset(y, 0, sizeof(y[0]) * is_begins[nw]);    
    for(d = 0; d < x->d; d += d_step) {
      for(i = 0; i < nw; i++) {
        const float *w = ws + i * x->d + d; 
        for(j = is_begins[i]; j < is_begins[i + 1]; j++) 
          y[j] += x_matrix_dotprod_d_slice(x, is[j], w, d, d_step); 
      }
    }
  }    

}

void x_matrix_addto_sparse(const x_matrix_t *x,
                           const x_matrix_sparse_t *a,
                           const float *betas,
                           float *ws, 
                           int nw, 
                           int d_step) {

  
  const int *is = a->ir, *is_begins = a->jc; 
  float *y = a->pr;
  
  int i, j;   
  if(x->d % d_step != 0) {
    for(i = 0; i < nw; i++) {
      float *w = ws + i * x->d; 
      vec_scale(w, x->d, betas[i]); 
      for(j = is_begins[i]; j < is_begins[i + 1]; j++) 
        x_matrix_addto(x, is[j], y[j], w);
    }
  } else {
    int d;
    for(d = 0; d < x->d; d += d_step) {
      for(i = 0; i < nw; i++) {
        float *w = ws + i * x->d + d; 
        vec_scale(w, d_step, betas[i]); 
        for(j = is_begins[i]; j < is_begins[i + 1]; j++) 
          x_matrix_addto_d_slice(x, is[j], y[j], w, d, d_step);
      }
    }
  }
}

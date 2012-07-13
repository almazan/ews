#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <sys/time.h>

#include <omp.h>

#include "jsgd.h"








static long jrandom_l(jsgd_params_t *params, long n) {
  long res; 
  res = rand_r(&params->rand_state); 
  res ^= (long)rand_r(&params->rand_state) << 31; 
  return res % n;
}


/*******************************************************************
 * Higher-level ops
 *
 */



static void renorm_w_factor(float *w, int d, float *fw) {
  if(1e-4 < *fw && *fw < 1e4) return;
  vec_scale(w, d, *fw);                    
  *fw = 1;
}

static void renorm_w_matrix(float *w, int d, int nclass, float *w_factors) {
  int j; 
  for(j = 0; j < nclass; j++) {
    vec_scale(w + d * j, d, w_factors[j]);
    w_factors[j] = 1.0;
  }
}

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))
#define NEWAC(type,n) (type*)calloc(sizeof(type),(n))
#define NEW(type) NEWA(type,1)


void jsgd_compute_scores(const x_matrix_t *x, int nclass, 
                         const float *w, const float *bias, 
                         float bias_term, 
                         int threaded, 
                         float *scores) {

  if(threaded) 
    x_matrix_matmul_thread(x, w, nclass, scores);    
  else
    x_matrix_matmul(x, w, nclass, scores);

  int i,j;
  for(i = 0; i < x->n; i++) {
    float *scores_i = scores + i * nclass;
    for(j = 0; j < nclass; j++) 
      scores_i[j] += bias[j] * bias_term;
  }

}
  

static double compute_accuracy(const x_matrix_t *x, const int *y, int nclass, 
                               const float *w, const float *bias, float bias_term, 
                               int threaded, 
                               int eval_criterion, double eval_exp_coeff) {
  float *scores = NEWA(float, nclass * x->n); 
  
  jsgd_compute_scores(x, nclass, w, bias, bias_term, threaded, scores);
  
  int i, j; 
  double accu = 0; 
  for(i = 0; i < x->n; i++) {
    const float *scores_i = scores + i * nclass;
    
    /* find rank of correct class */
    int nabove = 0, neq = 0; 
    float class_score = scores_i[y[i]];

    for(j = 0; j < nclass; j++) {      
      float score = scores_i[j];
      if(score > class_score) nabove ++; 
      else if(score == class_score) neq ++; 
    }
    int rank = nabove + neq / 2; /* a synthetic rank */
    
    switch(eval_criterion) {
    case JSGD_EVAL_TOP1: if(rank == 0) accu += 1; break;
    case JSGD_EVAL_MAP: accu += 1.0 / (rank + 1); break;
    case JSGD_EVAL_EXP: accu += exp(-rank * eval_exp_coeff); break;
    }    

  }
  free(scores);

  return accu / x->n;
}




/*******************************************************************
 * reporting
 *
 */


static double getmillisecs() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return tv.tv_sec*1e3 +tv.tv_usec*1e-3;
}


static double evaluation(int nclass,
                       const x_matrix_t *x,
                       const int *y,
                       float *w,
                       float *bias, 
                       jsgd_params_t *params, 
                       long t) {

  double t0 = getmillisecs(); 
  double elapsed_t = (t0 - params->t0) * 1e-3;
  double train_accuracy = -1, valid_accuracy = -1; 
  
  if(params->verbose)
    printf("Evaluation at epoch %ld (%ld samples, %.3f s), %ld dot prods, %ld W modifications\n",
           t / x->n, t, elapsed_t, params->ndp, params->nmodif);
  
  if(params->compute_train_accuracy) {
    train_accuracy = compute_accuracy(x, y, nclass, w, bias, params->bias_term, params->n_thread, 
                                      params->eval_criterion, params->eval_exp_coeff); 
    if(params->verbose) printf("train_accuracy = %.5f\n", train_accuracy); 
  }
  
  if(params->valid) {
    valid_accuracy = compute_accuracy(params->valid, params->valid_labels, nclass, w, bias, params->bias_term, 
                                      params->n_thread, params->eval_criterion, params->eval_exp_coeff); 
    
    if(params->verbose) printf("valid_accuracy = %.5f\n", valid_accuracy); 

  }      

  params->t_eval += (getmillisecs() - t0) * 1e-3; 

  long se = t / (x->n * params->eval_freq); 
  if(se < params->na_stat_tables) {
    if(params->valid_accuracies) params->valid_accuracies[se] = valid_accuracy;              
    if(params->train_accuracies) params->train_accuracies[se] = train_accuracy;              
    if(params->times) params->times[se] = elapsed_t;      
    if(params->ndotprods) params->ndotprods[se] = params->ndp; 
    if(params->nmodifs) params->nmodifs[se] = params->nmodif;       
  }  

  return valid_accuracy; 
}




void jsgd_params_set_default(jsgd_params_t *params) {
  memset(params, 0, sizeof(jsgd_params_t)); /* most defaults are ok at 0 */

  params->algo = JSGD_ALGO_OVR; 
  params->stop_valid_threshold = -1; 
  params->n_epoch = 10;
  
  params->lambda = 1e-4; 
  params->bias_term = 0.1;
  params->eta0 = 0.1;
  params->beta = 4;

  params->eval_freq = 10; 
  
  params->t_block = 16; 
  params->n_wstep = 32; 
  params->d_step = 1024;
}


/*******************************************************************
 * Learning, simple version
 *
 */


static void learn_epoch_ovr(int nclass,
                            const x_matrix_t *x,
                            const int *y,
                            float *w, 
                            float *bias, 
                            jsgd_params_t *params, 
                            long t0); 

static void compute_self_dotprods(const x_matrix_t *x, jsgd_params_t *params); 

static void learn_slice_others(int nclass,
                               const x_matrix_t *x,
                               const int *y,
                               float *w, 
                               float *bias, 
                               jsgd_params_t *params, 
                               long t0, long t1); 


static void find_eta0(int nclass,
                      const x_matrix_t *x,
                      const int *y,
                      float *w, 
                      float *bias, 
                      jsgd_params_t *params);

int jsgd_train(int nclass,
               const x_matrix_t *x,
               const int *y,
               float *w, 
               float *bias, 
               jsgd_params_t *params) {

  const long n = x->n, d = x->d; 

  params->rand_state = params->random_seed;
  
  float *best_w = NULL, *best_bias = NULL;

  if(params->valid) { /* we don't necessarily keep the last result */
    /* make temp buffers for current result */    
    best_w = w;  w = NEWA(float, nclass * d); 
    best_bias = bias;  bias = NEWA(float, nclass);
  }

  if(params->verbose) {
    printf("jsgd_learn: %ld training (%s) examples in %ldD from %d classes\n", 
           n, 
           params->algo ==  JSGD_ALGO_OVR ? "OVR" : 
           params->algo ==  JSGD_ALGO_MUL ? "MUL" : 
           params->algo ==  JSGD_ALGO_RNK ? "RNK" : 
           params->algo ==  JSGD_ALGO_WAR ? "WAR" : 
           "!! unknown",
           d, nclass); 
    if(params->valid) 
      printf("  validation on %ld examples every %d epochs (criterion: %s)\n", 
             params->valid->n, params->eval_freq, 
             params->eval_criterion == JSGD_EVAL_TOP1 ? "TOP1" : 
             params->eval_criterion == JSGD_EVAL_MAP ? "MAP" : 
             params->eval_criterion == JSGD_EVAL_EXP ? "EXP" : "!!unknown"); 

    if(params->valid && params->stop_valid_threshold > -1) 
      printf("stopping criterion: after %ld epochs or validation accuracy below best + %g\n",
             params->n_epoch, params->stop_valid_threshold);
    else
      printf("stopping after %ld epochs\n", params->n_epoch);
      
  } 

  params->ndp = params->nmodif = 0; 
  params->t0 = getmillisecs();
  params->best_valid_accuracy = 0;

  if(!params->use_input_w) {
    memset(w, 0, sizeof(w[0]) * nclass * d);
    memset(bias, 0, sizeof(bias[0]) * nclass);
  }

  if(params->algo == JSGD_ALGO_OVR && params->beta == 0) {
    params->beta = (int)sqrt(nclass); 
    if(params->verbose) printf("no beta provided, setting beta = %d\n", params->beta); 
  }

  if(params->eta0 == 0) {
    if(params->verbose) printf("no eta0 provided, searching...\n"); 
    find_eta0(nclass, x, y, w, bias, params); 
    if(params->verbose) printf("keeping eta0 = %g\n", params->eta0); 
  }

  if(params->algo  == JSGD_ALGO_OVR && params->use_self_dotprods) {
    compute_self_dotprods(x, params);     
  }

  long t; 
  int eval_valid = 0; /* is the current evaluation up-to-date? */

  for(t = 0; t < params->n_epoch * n; t += n) {

    if((t / n) % params->eval_freq == 0) {
      double valid_accuracy = evaluation(nclass, x, y, w, bias, params, t); 
      eval_valid = 1;

      if(valid_accuracy < params->best_valid_accuracy + params->stop_valid_threshold) {
        if(params->verbose)
          printf("not improving enough on validation: stop\n"); 
        break;
      }          

      if(valid_accuracy > params->best_valid_accuracy) {
        memcpy(best_w, w, sizeof(w[0]) * nclass * d);
        memcpy(best_bias, bias, sizeof(bias[0]) * nclass);        
        params->best_epoch = t / n; 
        params->best_valid_accuracy = valid_accuracy;
      }      
    }

    if(params->verbose > 1) {
      printf("Epoch %ld (%.3f s), %ld dot prods, %ld W modifications\r",
             t / x->n, (getmillisecs() - params->t0) * 1e-3, 
             params->ndp, params->nmodif);
      fflush(stdout);
    }

    switch(params->algo) {
      
    case JSGD_ALGO_OVR: 
      learn_epoch_ovr(nclass, x, y, w, bias, params, t);
      break;
      
    case JSGD_ALGO_MUL: case JSGD_ALGO_RNK: case JSGD_ALGO_WAR: 
      learn_slice_others(nclass, x, y, w, bias, params, t, t + n);
      break;
      
    default:       
      assert(!"not implemented");    
    }
    eval_valid = 0;
  }


  if(!eval_valid) {  /* one last evaluation */
    double valid_accuracy = evaluation(nclass, x, y, w, bias, params, t);     
    if(valid_accuracy > params->best_valid_accuracy) {
      memcpy(best_w, w, sizeof(w[0]) * nclass * d);
      memcpy(best_bias, bias, sizeof(bias[0]) * nclass);        
      params->best_epoch = t / n; 
      params->best_valid_accuracy = valid_accuracy;
    }  
  }

  if(params->verbose) 
    printf("returning W obtained at epoch %d (valid_accuracy = %g)\n", 
           params->best_epoch, params->best_valid_accuracy); 
  params->niter = t; 

  free(params->self_dotprods);
  if(best_w) {free(w); free(bias); }  

  return t;
}

/*******************************************************************
 * One Versus Rest implementation
 *
 */


/* 1 OVR step: update all w's from 1 vector */
static void learn_ovr_step(int nclass,
                           const x_matrix_t *x,
                           const int *y,
                           float *w, 
                           float *bias, 
                           jsgd_params_t *params, 
                           long t, float *w_factors, 
                           const int *ybars, int nybar) {

  const long n = x->n, d = x->d; 

  float bias_term = params->bias_term;
  
  long i = t % n; 
  int yi = y[i];
  
  double eta = params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
  double fw = 1 - eta * params->lambda;    
  
  if(params->verbose > 2) printf("iteration %ld, sample %ld, label %d, eta = %g\n", t, i, yi, eta);
  
  int k;
  for(k = 0; k < nybar; k++) {
    int ybar = ybars[k];     
    float *w_ybar = w + ybar * d; 
    
    float sense = y[i] == ybar ? 1.0 : -1.0; 
    
    double score = x_matrix_dotprod(x, i, w_ybar) * w_factors[ybar] + bias_term * bias[ybar];
    w_factors[ybar] *= fw;  
    
    if(sense * score < 1) { /* inside margin or on wrong side */
      x_matrix_addto(x, i, eta * sense / w_factors[ybar], w_ybar); 
      bias[ybar] += eta * sense * bias_term;
      params->nmodif++; 
    }
    
    renorm_w_factor(w_ybar, d, &w_factors[ybar]);  
  }
        
  params->ndp += nybar;

}


static void learn_epoch_ovr_blocks(int nclass,
                                   const x_matrix_t *x,
                                   const int *y,
                                   float *w, 
                                   float *bias, 
                                   jsgd_params_t *params, 
                                   long t0, 
                                   const int *ybars, int ldybar); 



static void learn_epoch_ovr(int nclass,
                            const x_matrix_t *x,
                            const int *y,
                            float *w, 
                            float *bias, 
                            jsgd_params_t *params, 
                            long t0) {

  assert(params->beta <= nclass - 1);    /* else  infinite loop */
  
  const long n = x->n, d = x->d; 
  int i, k, l;

  /* we choose in advance which elements to sample. This makes it
     possible to reproduce results exactly across implementations */

  int ldybar = params->beta + 1; 
  int *ybars = NEWA(int, ldybar * n);
  
  long t, t1 = t0 + n;
  for(t = t0; t < t1; t++) {   
    long i = t % n; 
    int yi = y[i];

    int *seen = ybars + ldybar * (t - t0); 
    
    /* draw beta ybars */
    seen[0] = yi; 
    for(k = 1; k <= params->beta; k++) {
      int ybar; 
      
      /* sample a label different from the ones seen so far. 
         TODO: improve speed when not beta << nclass */
    draw_ybar:
      ybar = jrandom_l(params, nclass); 
      for(l = 0; l < k; l++) if(ybar == seen[l]) goto draw_ybar;          
      seen[k] = ybar; 
    }    
    
  }

  double tt0 = getmillisecs(); 

  /* W is encoded as W * diag(w_factors). This avoids some vector
     multiplications */
  float *w_factors = NEWA(float, nclass); 

  if(!params->n_thread) {
  
    for(i = 0; i < nclass; i++) w_factors[i] = 1; 
   
    for(t = t0; t < t1; t++) {
      int *ybars_i = ybars + ldybar * (t - t0);       
      learn_ovr_step(nclass, x, y, w, bias, params, t, w_factors, ybars_i, ldybar);       
    }
    renorm_w_matrix(w, d, nclass, w_factors);

  } else 
    learn_epoch_ovr_blocks(nclass, x, y, w, bias, params, t0, ybars, ldybar); 
 
  if(params->verbose > 2) printf("OVR epoch t = %.3f ms\n", getmillisecs() - tt0);

  free(w_factors); 
  free(ybars);
}


/***************************************************************************
 * blocked code (hairy!)
 */


/* 1 OVR step (transposed): all w's from a set of vectors */
static void learn_ovr_step_w(int nclass,
                             const x_matrix_t *x,
                             const int *y,
                             float *w_yi, 
                             float *bias_yi_io, 
                             const jsgd_params_t *params, 
                             int yi,
                             long t0, const int * ts, int nt, 
                             int *ndp_io, int *nmodif_io) {
  
  const long n = x->n, d = x->d; 

  float bias_term = params->bias_term;

  if(params->verbose > 2) printf("   handling W, label %d, %d ts\n", yi, nt);

  float bias_yi = *bias_yi_io; 
  float w_factor = 1.0;
  int nmodif = 0; 

  int k;
  for(k = 0; k < nt; k++) {
    long t = t0 + ts[k];
    long i = t % n; 
    float sense = y[i] == yi ? 1.0 : -1.0;

    double eta = params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    double fw = 1 - eta * params->lambda;    

    double score = x_matrix_dotprod(x, i, w_yi) * w_factor + bias_term * bias_yi; 

    if(params->verbose > 3) printf("      score with x_%ld = %g (%d modifs)\n", i, score, nmodif);
  
    w_factor *= fw;     

    if(sense * score < 1) {
      x_matrix_addto(x, i, sense * eta / w_factor, w_yi); 
      bias_yi += sense * eta * bias_term;
      nmodif ++; 
    }    

    renorm_w_factor(w_yi, d, &w_factor); 
  }

  vec_scale(w_yi, d, w_factor); 

  *bias_yi_io = bias_yi; 
  *ndp_io += nt; 
  *nmodif_io += nmodif; 
}

  


/* same, all dot products are precomputed */
static void learn_ovr_step_w_dps(int nclass,
                                 const x_matrix_t *x,
                                 const int *y,
                                 float *w_yi, 
                                 float *bias_yi_io, 
                                 const jsgd_params_t *params, 
                                 int yi,
                                 long t0, const int * ts, int nt, 
                                 const float *w_dps, const float *self_dps, int ldsd,
                                 int *correction_is, float *correction_terms, 
                                 float *w_factor_out, 
                                 int *ndp_io, int *nmodif_io) {
  
  const long n = x->n; 

  float bias_term = params->bias_term;

  if(params->verbose > 2) printf("  DPS handling W, label %d, %d ts\n", yi, nt);

  float bias_yi = *bias_yi_io; 
  float w_factor = 1.0;
  int nmodif = 0; 
  
  int i0 = t0 % n;
  int k;
  for(k = 0; k < nt; k++) {
    long t = t0 + ts[k];
    long i = t % n; 
    float sense = y[i] == yi ? 1.0 : -1.0;

    double eta = params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    double fw = 1 - eta * params->lambda;    

    /* reconstruct score */
    double score = w_dps[k];
    int c;
    for(c = 0; c < nmodif; c++) 
      score += self_dps[(correction_is[c] - i0) + ldsd * (i - i0)] * correction_terms[c];

    score = score * w_factor + bias_term * bias_yi; 

    assert(score == 0 || fabs(score) > 1e-20); 
    assert(fabs(score) < 1e10); 

    if(params->verbose > 3) printf("      score with x_%ld = %g (%d modifs)\n", i, score, nmodif);
  
    w_factor *= fw;     

    if(sense * score < 1) {
      correction_is[nmodif] = i; 
      correction_terms[nmodif] = sense * eta / w_factor;
      bias_yi += sense * eta * bias_term;
      nmodif ++; 
    }    

  }

  *w_factor_out = w_factor; 
  *bias_yi_io = bias_yi; 
  *ndp_io += nt; 
  *nmodif_io += nmodif; 
}




/* 
 * input: 
 * 
 * vals(n, m)   st. 0 <= v(i, j) < nval
 *
 * output: 
 *
 * js(m * n) 
 * begins(nval + 1) 
 *
 * j in js(begins(w): begins(w + 1)) iff w in vals(:, j)
 * 
 */

static void make_histogram_with_refs(int n, int m, 
                                     const int *vals, 
                                     int nval,                           
                                     int *js,
                                     int *begins) {
  memset(begins, 0, (nval + 1) * sizeof(float)); 
  begins++; 
  int i, j;

  /* first make histogram */
  for(i = 0; i < m * n; i++) 
    begins[vals[i]] ++; 

  /* cumulative sum */
  int accu = 0;
  for(i = 0; i < nval; i++) {
    int b = begins[i];
    begins[i] = accu; 
    accu += b; 
  }
  assert(accu == m * n); 
  /* now begins[i] contains offset in js where to write values */  

  const int *vp = vals;
  for(j = 0; j < m ; j++) 
    for(i = 0; i < n; i++) 
      js[begins[*vp++]++] = j; 

  /* now all values are written so begins[v] contains end of segment
     v, but we did begins++ so the orginial array indeed contains the
     beginning */
  assert(begins[nval - 1] == n * m);   
}



static void learn_epoch_ovr_blocks(int nclass,
                                   const x_matrix_t *x,
                                   const int *y,
                                   float *w, 
                                   float *bias, 
                                   jsgd_params_t *params, 
                                   long t0, 
                                   const int *ybars, int ldybar) {
  const long n = x->n, d = x->d; 
  long t_block = params->t_block;   
  int ngroup = (n + params->t_block - 1) / params->t_block; 
  long ldg = params->t_block * params->t_block; 


  double tt0 = getmillisecs(); 
  

  /* convert the ybar arrays to arrays that are indexed by W */
  
  int *all_ts = NEWA(int, (ldybar * t_block) * ngroup);
  int *all_ts_begins = NEWAC(int, (nclass + 1) * ngroup);
  
  long ti; 
  for(ti = 0; ti < ngroup; ti++) {
    int i0 = ti * params->t_block; 
    int i1 = (ti + 1) * params->t_block; 
    if(i1 > n) i1 = n; 
    int *ts = all_ts + ti * (ldybar * t_block); 
    int *ts_begins = all_ts_begins + ti * (nclass + 1); 
    
    make_histogram_with_refs(ldybar, i1 - i0, ybars + ldybar * ti * t_block, nclass, 
                             ts, ts_begins); 

    /* for ts_begins[j] <= i < ts_begins[j + 1], 
       
       x(:, t0 + ts[i]) interacts with W(:, j)
       
    */
    
  }     

  int n_wstep = params->n_wstep; 
  if(!n_wstep) 
    n_wstep = omp_get_max_threads();    


  int ndp = 0, nmodif = 0; 
  
  int wi; 
#pragma omp parallel for  reduction(+: ndp, nmodif)
  for(wi = 0; wi < n_wstep; wi++) {
    int wa = wi * nclass / n_wstep, wb = (wi + 1) * nclass / n_wstep;
    
    int ti; 
    for(ti = 0; ti < ngroup; ti++) {
      int i0 = ti * params->t_block; 
      int i1 = (ti + 1) * params->t_block; 
      if(i1 > n) i1 = n; 

      int *ts = all_ts + ti * (ldybar * t_block); 
      int *ts_begins = all_ts_begins + ti * (nclass + 1); 
      
      int nnz = ts_begins[wb] - ts_begins[wa]; 
      
      if(params->verbose > 2) 
        printf("handling block W %d:%d * T %d:%d, %d interactions (%.2f %%)\n", 
               wa, wb, i0, i1, nnz, nnz * 100.0 / ((wb - wa) * (i1 - i0))); 

      
      if(!params->use_self_dotprods) {
      
        int yi; 
        for(yi = wa; yi < wb; yi++) 
          learn_ovr_step_w(nclass, x, y, w + yi * d, &bias[yi], 
                           params, yi, t0 + i0, 
                           ts + ts_begins[yi], 
                           ts_begins[yi + 1] - ts_begins[yi], 
                           &ndp, &nmodif);      
        
      } else {
          
        /* dot products with W */
        x_matrix_sparse_t w_dps; 
        
        x_matrix_sparse_init(&w_dps, i1 - i0, wb - wa, nnz); 
        
        int k = 0, yi, i;
        for(yi = wa; yi < wb; yi++) {        
          w_dps.jc[yi - wa] = k;
          for(i = ts_begins[yi]; i < ts_begins[yi + 1]; i++) 
            w_dps.ir[k++] = ts[i] + i0;               
        }
        w_dps.jc[wb - wa] = k;
        
        x_matrix_matmul_subset(x, &w_dps, w + wa * d, wb - wa, params->d_step); 
        
        /* dot products within x */
        float *self_dps = params->self_dotprods + ti * ldg;
        
        /* W update */
        x_matrix_sparse_t correction; 
        x_matrix_sparse_init(&correction, i1 - i0, wb - wa, nnz);           
        float *w_factors = NEWA(float, wb - wa); 
        
        /* play OVR */
        int nmodif_i = 0;
        for(yi = wa; yi < wb; yi++) {
          correction.jc[yi - wa] = nmodif_i;             
          learn_ovr_step_w_dps(nclass, x, y, w + yi * d, &bias[yi], 
                               params, yi, t0 + i0, 
                               ts + ts_begins[yi], 
                               ts_begins[yi + 1] - ts_begins[yi], 
                               w_dps.pr + w_dps.jc[yi - wa],                                  
                               self_dps, i1 - i0, 
                               correction.ir + nmodif_i, 
                               correction.pr + nmodif_i, 
                               &w_factors[yi - wa], 
                               &ndp, &nmodif_i);      
        }
        correction.jc[wb - wa] = nmodif_i; 
        nmodif += nmodif_i; 
        if(params->verbose > 2) 
          printf("applying %d W corrections\n", nmodif_i); 
        
        /* apply correction to Ws */          
        x_matrix_addto_sparse(x, &correction, w_factors, w + wa * d, wb - wa, params->d_step);
        
        free(w_factors); 
        
        x_matrix_sparse_clear(&w_dps); 
        x_matrix_sparse_clear(&correction); 
        
      }
      
    }
  }
  free(all_ts); 
  free(all_ts_begins); 

  params->ndp += ndp; 
  params->nmodif += nmodif; 

  if(params->verbose > 2) printf("blocked t = %.3f ms\n", getmillisecs() - tt0);

}



static void compute_self_dotprods(const x_matrix_t *x, jsgd_params_t *params) {
  int n = x->n; 
  int ngroup = (n + params->t_block - 1) / params->t_block; 
  long ldg = params->t_block * params->t_block; 

  if(params->verbose) 
    printf("Computing dot products within training data (%d groups of max size %d) using %.2f MB...\n", 
           ngroup, params->t_block, ngroup * ldg * 4 / (1024.0*1024));        
  
  params->self_dotprods = NEWA(float, ngroup * ldg); 
  
  double t0 = getmillisecs(); 

  int i; 
#pragma omp parallel for 
  for(i = 0; i < ngroup; i++) {
    int i0 = i * params->t_block; 
    int i1 = (i + 1) * params->t_block; 
    if(i1 > n) i1 = n; 
    x_matrix_matmul_self(x, i0, i1, params->self_dotprods + i * ldg); 
  }
  
  if(params->verbose) 
    printf("done in %.3f s\n", (getmillisecs() - t0) * 1e-3); 

}


/*******************************************************************
 * Other algorithms (not OVR) implementation
 *
 */


static void learn_slice_others(int nclass,
                            const x_matrix_t *x,
                            const int *y,
                            float *w, 
                            float *bias, 
                            jsgd_params_t *params, 
                            long t0, long t1) {

  
  const long n = x->n, d = x->d; 
  float bias_term = params->bias_term;
  float *scores = NULL; 


  int *perm = NULL; 
  float *lk = NULL; 
  
  if(params->algo == JSGD_ALGO_MUL) 
    scores = NEWA(float, nclass);

  if(params->algo == JSGD_ALGO_WAR) {
    perm = NEWA(int, nclass); 
    lk = NEWA(float, nclass); 
    float accu = 0; 
    int k; 
    for(k = 0; k < nclass; k++) {
      accu += 1.0 / (1 + k); 
      lk[k] = accu;
    }
  }

  float w_factor = 1.0;

  long t;
  for(t = t0; t < t1; t++) {

    long i = t % n; 
    int yi = y[i];
    float *w_yi = w + d * yi; 
    
    double eta =  params->fixed_eta ? params->eta0 : params->eta0 / (1 + params->lambda * params->eta0 * t);
    double fw = 1 - eta * params->lambda;    

    if(params->verbose > 2) printf("iteration %ld, sample %ld, label %d, eta = %g\n", t, i, yi, eta);

    int ybar = -1; 

    if(params->algo == JSGD_ALGO_MUL) {

      /* find worst violation */
      x_matrix_matmul_slice(x, i, i + 1, w, nclass, scores); 
      params->ndp += nclass; 
      
      int j; 
      float best_score = -1e20;    
      for(j = 0; j < nclass; j++) {
        float score = scores[j] * w_factor + bias[j] * bias_term;
        
        if(j != yi) score += 1;
        
        if(score > best_score) {
          best_score = score; 
          ybar = j;       
        }
      }
    } else if(params->algo == JSGD_ALGO_RNK) {
      double score_yi = x_matrix_dotprod(x, i, w_yi) * w_factor + bias_term * bias[yi]; 

      do {
        ybar = jrandom_l(params, nclass); 
      } while(ybar == yi);
      
      float *w_ybar = w + d * ybar; 
      double score_ybar = x_matrix_dotprod(x, i, w_ybar) * w_factor + bias_term * bias[ybar]; 

      double L_tri = 1 - score_yi + score_ybar;
      
      if(!(L_tri > 0)) ybar = yi; /* do nothing */

      params->ndp += 2; 
    } else if(params->algo == JSGD_ALGO_WAR) {
      double score_yi = x_matrix_dotprod(x, i, w_yi) * w_factor + bias_term * bias[yi]; 
      
      int j, k; 
      for(j = 0; j < nclass - 1; j++) perm[j] = j; 
      perm[yi] = nclass - 1;

      for(k = 0; k < nclass - 1; k++) {
        int k2 = jrandom_l(params, nclass - 1 - k) + k; 
        ybar = perm[k2]; 
        perm[k2] = perm[k]; 
        /* perm[k] = ybar; */
        
        float *w_ybar = w + d * ybar; 
        double score_ybar = x_matrix_dotprod(x, i, w_ybar) * w_factor + bias_term * bias[ybar]; 
        
        double L_tri = 1 - score_yi + score_ybar;
        
        if(L_tri > 0) break;
      }
      params->ndp += k; 
      
      if(k == nclass - 1) {
        ybar = yi; /* did not find a violation */
      } else {
        /* recompute eta */
        eta *= lk[(nclass - 1) / (k + 1)]; 
      }
    }

    w_factor *= fw;

    if(ybar != yi) {
      x_matrix_addto(x, i, eta / w_factor, w_yi); 
      bias[yi] += eta * bias_term;

      float *w_ybar = w + d * ybar; 
      x_matrix_addto(x, i, -eta / w_factor, w_ybar); 
      bias[ybar] -= eta * bias_term;

      params->nmodif += 2;              
    }
    
    if(w_factor < 1e-4) {
      vec_scale(w, d * nclass, w_factor); 
      w_factor = 1.0;
    }

  }
  vec_scale(w, d * nclass, w_factor); 
  
  free(perm); 
  free(scores); 
  free(lk);
}



/*******************************************************************
 * Searching eta0 
 *
 */



/* per-example loss */
static void compute_losses(const x_matrix_t *x, const int *y, int nclass, 
                            const float *w, const float *bias, 
                            const jsgd_params_t *params, 
                            float *losses) {
  float *scores = NEWA(float, nclass * x->n);   
  jsgd_compute_scores(x, nclass, w, bias, params->bias_term, params->n_thread, scores);

  int n = x->n;
  int i, j; 
  
  switch(params->algo) {
  case JSGD_ALGO_OVR: 
    {    /* sum 1v1 hinge losses */
      int *hist = NEWAC(int, nclass);       
      for(i = 0; i < n; i++) hist[y[i]]++;
      
      float rho = 1.0 / (1 + params->beta); 

      memset(losses, 0, sizeof(float) * n); 

      for(j = 0; j < nclass; j++) {
        for(i = 0; i < n; i++) {          
          float yi, weight; 

          if(y[i] == j) { 
            yi = 1; 
            weight = rho / hist[j]; /* eq(3) of paper */
          } else {
            yi = -1; 
            weight = (1 - rho) / (n - hist[j]); 
          }

          float L_ovr = 1 - yi * scores[i * nclass + j];
          if(L_ovr < 0) L_ovr = 0; 
          
          losses[i] += weight * L_ovr;
        }
      }
      free(hist); 
    }
    break; 
    
  case JSGD_ALGO_MUL: 
    
    for(i = 0; i < n; i++) {
      double vmax = -10000; 
      for(j = 0; j < nclass; j++) {
        float delta = j == y[i] ? 0.0 : 1.0;
        float v = delta + scores[i * nclass + j]; 
        if(v > vmax) vmax = v;         
      }
      losses[i] = vmax - scores[i * nclass + y[i]]; 
    }
    break; 
    
    case JSGD_ALGO_RNK: 
      for(i = 0; i < n; i++) {
        float loss = 0; 
      for(j = 0; j < nclass; j++) {
        float delta = j == y[i] ? 0.0 : 1.0;
        float L_tri = delta + scores[i * nclass + j] - scores[i * nclass + y[i]]; 
        if(L_tri < 0) L_tri = 0; 
        loss += L_tri; 
      }
      losses[i] = loss; 
    }
    break; 

  default: 
    assert(!"not implemented");     

  }
  
  free(scores); 
}


/* cost according to eq(1) of paper */

static float compute_cost(const x_matrix_t *x, const int *y, int nclass, 
                         const float *w, const float *bias, 
                         const jsgd_params_t *params) {

  {
    int i;
    for(i = 0; i < x->n; i++) assert(y[i] >= 0 && y[i] < nclass); 
  }

  float *losses = NEWA(float, x->n);   
  compute_losses(x, y, nclass, w, bias, params, losses); 

  float loss = 0; 
  int i; 
  for(i = 0; i < x->n; i++) loss += losses[i];

  /* regularization term */
  double sqnorm_W = vec_sqnorm(w, nclass * x->d) + 
    vec_sqnorm(bias, nclass) * params->bias_term * params->bias_term;

  return loss + 0.5 * params->lambda * sqnorm_W; 
}

static void *memdup(void *a, size_t size) {
  return memcpy(malloc(size), a, size); 
}

static float eval_eta(int nsubset, float eta0, 
                      int nclass,
                      const x_matrix_t *x,
                      const int *y,
                      float *w_in, 
                      float *bias_in, 
                      jsgd_params_t *params) {
  jsgd_params_t subset_params = *params;
  x_matrix_t subset_x = *x; 
  
  subset_x.n = nsubset; 
  subset_params.eta0 = eta0;
  

  float *w = memdup(w_in, sizeof(float) * nclass * x->d); 
  float *bias = memdup(bias_in, sizeof(float) * nclass); 
  
  switch(params->algo) {
    
  case JSGD_ALGO_OVR: 
    learn_epoch_ovr(nclass, x, y, w, bias, &subset_params, 0);
    break;
    
  case JSGD_ALGO_MUL: case JSGD_ALGO_RNK: case JSGD_ALGO_WAR: 
    learn_slice_others(nclass, x, y, w, bias, &subset_params, 0, nsubset);
    break;
    
  default:       
    assert(!"not implemented");    
  }

  float cost = compute_cost(&subset_x, y, nclass, w, bias, &subset_params); 

  free(w); 
  free(bias);
  
  return cost; 
}

static void find_eta0(int nclass,
                      const x_matrix_t *x,
                      const int *y,
                      float *w, 
                      float *bias, 
                      jsgd_params_t *params) {

  int nsubset = 1000; /* evaluate on this subset */
  if(nsubset > x->n) nsubset = x->n; 

  float factor = 2;   /* multiply or divide eta by this */

  float eta1 = 1;     /* intial estimate */ 
  float cost1 = eval_eta(nsubset, eta1, nclass, x, y, w, bias, params); 


  float eta2 = eta1 * factor; 
  float cost2 = eval_eta(nsubset, eta2, nclass, x, y, w, bias, params); 

  if(params->verbose > 1) printf("  eta1 = %g cost1 = %g, eta2 = %g cost2 = %g\n", eta1, cost1, eta2, cost2); 

  if(cost2 > cost1) {
    /* switch search direction */
    float tmp = eta1; eta1 = eta2; eta2 = tmp; 
    tmp = cost1; cost1 = cost2; cost2 = tmp; 
    factor = 1 / factor; 
  }

  /* step eta into search direction until cost increases */

  do {  
    eta1 = eta2; 
    eta2 = eta2 * factor;
    cost1 = cost2;     
    cost2 = eval_eta(nsubset, eta2, nclass, x, y, w, bias, params); 
    if(params->verbose > 1) printf("  eta2 = %g cost2 = %g\n", eta2, cost2); 
  } while(cost1 > cost2);
  
  /* keep smallest */
  params->eta0 = eta1 < eta2 ? eta1 : eta2;  

}

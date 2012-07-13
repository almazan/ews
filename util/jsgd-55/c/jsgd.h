#ifndef SGD_H_INCLUDED
#define SGD_H_INCLUDED

#include "x_matrix.h"



/* Parameters, statistics and temp data for the learning
 * algorithm. Most values can be kept at 0, see
 * jsgd_params_set_default for the remaining ones */

typedef struct {
  int verbose;     /* verbosity level */
  int n_thread;    /* number of threads to use (0 = never create threads) */
  int random_seed; /* running with the same random seed should give the same results */

  /* initialization */
  int use_input_w; /* use input w as initialization (else it is cleared) */
  
  enum {
    JSGD_ALGO_OVR,
    JSGD_ALGO_MUL,
    JSGD_ALGO_RNK,
    JSGD_ALGO_WAR,
  } algo;

  long n_epoch;            /* number of samples visited */
  
  /* validation set to decide stopping + report stats */
  x_matrix_t *valid;        /* validation matrix */
  int *valid_labels;        /* corresponding labels (0 based) */

  /* evaluation passes */

  enum {                    /* type of accuracy to compute: average of... */
    JSGD_EVAL_TOP1,         /* ... top-1 accuracy */
    JSGD_EVAL_MAP,          /* ... per class AP: 1/rank of correct class (less noisy ?) */
    JSGD_EVAL_EXP,          /* ... exp(-exp_coeff * rank) */
  } eval_criterion; 

  int eval_freq;               /* evaluate on validation set every that many epochs */
  int compute_train_accuracy;  /* (boolean) disabled by default, may be expensive */
  double stop_valid_threshold; /* stop if the accuracy is lower than
                                  the accuracy at the previous
                                  evaluation + this much */
  double eval_exp_coeff;       /* parameter for JSGD_EVAL_EXP */


  /* algorithm parameters */
  double lambda;            
  double bias_term;         /* this term is appended to each training vector */
  double eta0;              /* intial value of the gradient step */
  int beta;                 /* for OVR, how many negatives to sample per training vector */
  int fixed_eta;            /* (boolean) use a fixed step */

  /* output stats */  
  int best_epoch;           /* epoch at which the result was found */
  long niter;               /* effective number of iterations */
  long ndp, nmodif;         /* nb of dot products performed, number of W column modifications */
  double t_eval;            /* time spent on evaluation passes */

  /* stats stampled at each evaluation */
  long na_stat_tables;      /* sizes of following tables (allocated by caller) */
  double *times;            /* timestamp at each evaluation (seconds) */
  double *train_accuracies; 
  double *valid_accuracies; 
  int *ndotprods, *nmodifs; 


  /* internal parameters and temp data */
  double best_valid_accuracy;   /* best validation accuracy seen so far */
  double t0;                    /* begin time */
  unsigned int rand_state;      /* current random state */
  int t_block, n_wstep, d_step; /* blocking parameters */
  float *self_dotprods;         /* buffer for dot products between the x(:, i)'s */
  int use_self_dotprods;        /* should we use them? */

} jsgd_params_t; 



/* fill in decent default params (OVR) */
void jsgd_params_set_default(jsgd_params_t *params); 

/* Uses SGD to learn a classifier
 * labels should be in 0:nclass-1
 * 
 * @param nclass       nb of classes
 * @param x            matrix of train vectors, size (d, n)
 * @param y(n)         labels of each vector. Should be in 0:nclass-1
 * @param w(d,nclass)  output w matrix
 * @param bias(nclass) output bias coefficients
 * @param params       all other parameters 
 *
 * @return             nb of iterations performed
 */

int jsgd_train(int nclass,
               const x_matrix_t *x,
               const int *y,
               float * w, 
               float * bias, 
               jsgd_params_t *params); 


/* Compute classification scores for a matrix. 
 * 
 * @param nclass           nb of classes
 * @param x                matrix of vectors to classifiy, size (d, n)
 * @param w(d,nclass)      w matrix
 * @param bias(nclass)     bias coefficients
 * @param threaded         (boolean) whether to use threads
 * @param scores(nclass,n) output classification scores
 */

void jsgd_compute_scores(const x_matrix_t *x, int nclass, 
                         const float *w, const float *bias, 
                         float bias_term, 
                         int threaded, 
                         float *scores);

#endif

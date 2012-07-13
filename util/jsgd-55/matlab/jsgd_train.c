#include <stdio.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <mex.h>

#include "../c/jsgd.h"

static void mxArray_to_x_matrix(const mxArray *a, 
                               x_matrix_t *x) {

  if(mxGetClassID(a) == mxSINGLE_CLASS) {

    x->encoding = JSGD_X_FULL; 
    x->n = mxGetN (a);
    x->d = mxGetM (a);; 
    x->data = mxGetData(a); 
    
#if 0
  } else if(mxIsSparse(a)) {
    /* for some reason, there are no single sparse arrays */ 
    x->encoding = JSGD_X_SPARSE; 
    x->

#endif
  } else if(mxGetClassID(a) == mxSTRUCT_CLASS && 
            mxGetField(a, 0, "centroids")) {
    mxArray *pqfield = mxGetField(a, 0, "centroids"); 
    if(!pqfield || mxGetClassID(pqfield) != mxSINGLE_CLASS) 
      mexErrMsgTxt("centroids must be 3D single precision");
    
    const mwSize *dim = mxGetDimensions(pqfield);
    int ndim = mxGetNumberOfDimensions(pqfield); 
    
    if(ndim != 3 || dim[1] != 256) 
      mexErrMsgTxt("centroids must be 3D single precision, middle dim 256");
    
    x->encoding = JSGD_X_PQ;
    x->nsq = dim[2];
    x->d = dim[2] * dim[0];
    x->centroids = mxGetData(pqfield); 
    
    mxArray *codesfield = mxGetField(a, 0, "codes"); 
    if(!codesfield || mxGetClassID(codesfield) != mxUINT8_CLASS || mxGetM(codesfield) != x->nsq) 
      mexErrMsgTxt("codes must be unint8 and size compatible with centroids");
    
    x->codes = mxGetData(codesfield); 
    x->n = mxGetN (codesfield);
  } else 
    mexErrMsgTxt("need single precision array or struct for train vectors.");

  
}


#define IN_X prhs[0]
#define IN_Y prhs[1]
#define OUT_W plhs[0]
#define OUT_STATS plhs[1]

void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[]) {

  if (nrhs < 2 || nrhs % 2 != 0) 
    mexErrMsgTxt("even nb of input arguments required.");
  else if (!(nlhs == 1 || nlhs == 2)) 
    mexErrMsgTxt("1 or 2 outputs");

    
  /* training matrix */

  x_matrix_t x; 

  mxArray_to_x_matrix(IN_X, &x); 

  int n = x.n, d = x.d; 
  
  if(mxGetClassID(IN_Y) != mxINT32_CLASS) 
    mexErrMsgTxt("need int32 array for labels.");

  if(mxGetM (IN_Y) * mxGetN (IN_Y) != n) 
    mexErrMsgTxt("labels size not consistent with train matrix");

  /* algorithm parameters */

  jsgd_params_t params;
  jsgd_params_set_default(&params); 


  /* validation matrix */
  x_matrix_t valid;  
  valid.n = 0;
  
  int *valid_labels = NULL; 
  int n_valid_labels = -1;

  /* parse additional arguments */
  int i, j; 
  for(i = 2; i < nrhs; i += 2) {
    char varname[256];
    if (mxGetClassID(prhs[i]) != mxCHAR_CLASS || 
        mxGetString (prhs[i], varname, 256) != 0)
      mexErrMsgTxt ("invalid option string");         

#define GETOPT(type, name) if(!strcmp(varname, #name)) \
      params.name = (type) mxGetScalar (prhs[i+1]);

    GETOPT(int, n_epoch)
    else GETOPT(int, verbose)
    else GETOPT(int, eval_freq)
    else GETOPT(double, stop_valid_threshold)
    else GETOPT(double, lambda)
    else GETOPT(double, bias_term)
    else GETOPT(double, eta0)
    else GETOPT(int, beta)
    else GETOPT(int, random_seed)
    else GETOPT(int, fixed_eta)
    else GETOPT(int, n_thread)

    else if(!strcmp(varname, "algo")) {
      char algoname[256];
      if (mxGetClassID(prhs[i + 1]) != mxCHAR_CLASS || 
          mxGetString (prhs[i + 1], algoname, 256) != 0)
        mexErrMsgTxt ("invalid algo string");         
      
      params.algo = 
        !strcasecmp(algoname, "ovr") ? JSGD_ALGO_OVR :
        !strcasecmp(algoname, "mul") ? JSGD_ALGO_MUL :
        !strcasecmp(algoname, "rnk") ? JSGD_ALGO_RNK :
        !strcasecmp(algoname, "war") ? JSGD_ALGO_WAR :
        -1;
      
      if (params.algo == -1) mexErrMsgTxt ("invalid algo name");      

    } else if(!strcmp(varname, "valid")) {
      
      mxArray_to_x_matrix(prhs[i+1], &valid); 

    } else if(!strcmp(varname, "valid_labels")) {
      n_valid_labels = mxGetN (prhs[i+1]) * mxGetM (prhs[i+1]);
      if(mxGetClassID(prhs[i+1]) != mxINT32_CLASS) 
        mexErrMsgTxt ("valid labels not same dimension as train");
      valid_labels = mxMalloc(sizeof(int) * n_valid_labels); 
      const int *valid_labels_1 = mxGetData(prhs[i+1]); 
      for(j = 0; j < n_valid_labels; j++) 
        valid_labels[j] = valid_labels_1[j] - 1;       

    } else 
      mexErrMsgTxt("unknown variable name");
  }

  
  
  /* validation */
  if(valid_labels || valid.n > 0) {
    if(!(valid_labels && valid.n > 0 && n_valid_labels == valid.n && valid.d == x.d)) 
      mexErrMsgTxt("validation and validation_labels not in sync");
    params.valid = &valid;     
    params.valid_labels = valid_labels;
  }
   
  /* shift to 0-based labels & find nclass */
  int *labels_0 = mxMalloc(sizeof(int) * n); 
  int nclass = 0; 

  {
    const int *labels_1 = mxGetData(IN_Y);
      
    for(i = 0; i < n; i++) {
      if(labels_1[i] > nclass) nclass = labels_1[i];
      if(labels_1[i] <= 0) mexErrMsgTxt("labels should be > 0");
      labels_0[i] = labels_1[i] - 1;       
    }

    if(nclass > 100000) mexErrMsgTxt("too many classes?");
  }

  if(nclass < params.beta + 1) params.beta = nclass - 1;

  /* output */          
  float *bias = mxMalloc(sizeof(float) * nclass); 
  OUT_W = mxCreateNumericMatrix (d + 1, nclass, mxSINGLE_CLASS, mxREAL);
  float *w = mxGetData(OUT_W); 

  if(nlhs == 2) {
    OUT_STATS = mxCreateStructMatrix(1, 1, 0, NULL); 
    params.na_stat_tables = (params.n_epoch * n) / params.eval_freq + 1; 

    mxArray *ma;

#define STATFIELD(name, class)                                          \
    mxAddField(OUT_STATS, #name);                                       \
    ma = mxCreateNumericMatrix (params.na_stat_tables, 1, class, mxREAL); \
    params.name = mxGetData(ma);                                        \
    mxSetField(OUT_STATS, 0, #name, ma);  

    STATFIELD(valid_accuracies, mxDOUBLE_CLASS); 
    STATFIELD(times, mxDOUBLE_CLASS); 
    STATFIELD(train_accuracies, mxDOUBLE_CLASS); 
    STATFIELD(ndotprods, mxINT32_CLASS); 
    STATFIELD(nmodifs, mxINT32_CLASS); 

  }

  /* the call */
  jsgd_train(nclass, &x, labels_0, w, bias, &params); 
   
  /* delicate operation: w was used with a different stride in jsgd,
     redo properly and interleave with bias */
  for(i = nclass - 1; i >= 0; i--) {
    memmove(w + i * (d + 1), w + i * d, d * sizeof(w[0]));
    w[i * (d + 1) + d] = bias[i] * params.bias_term;
  }
  
  mxFree(bias); 
  mxFree(labels_0); 
  mxFree(valid_labels); 

}


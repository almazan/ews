=============================================================
==== README for JSGD
=============================================================



What is this?
=============

This a Stochastic Gradient Descent algorithm used to train linear
multiclass classifiers. It is biased towards large classification
problems (many classes, many examples, high dimensional data).

The reference paper is: 

@inproceedings{perronnin:hal-00690014, 
  AUTHOR = {Perronnin, Florent and Akata, Zeynep and Harchaoui, Zaid and Schmid, Cordelia}, 
  TITLE = {{Towards Good Practice in Large-Scale Learning for Image Classification}}, 
  BOOKTITLE = {{Computer Vision and Pattern Recognition}}, 
  YEAR = {2012}, 
  MONTH = Jun, 
  ADDRESS = {Providence, Rhode Island, United States}, 
  URL = {http://hal.inria.fr/hal-00690014} 
}

The package is also inspired by Leon Bottou's package, available at 

http://leon.bottou.org/projects/sgd

Files
=====

The directories in the package are:

ref/    contains a basic Matlab implementation of the SGD algorithm.

c/      contains the C implementation.

matlab/ contains the MEX wrappers for Octave/Matlab. There is a
        precompiled version for 64-bit Linux

python/ contains the Python wrappers, and the optimization code that
        uses cross-validation to find the best parameters of the
        method.

Compiling / requirements
========================

Requirements: 

- 64-bit Linux. The code should be reasonably portable, but no other
platform was tested.

- an implementation of the Basic Linear Algebra Subroutines (BLAS),
that are interfaced with the Fortran conventions.

- SSE vector operations. They are declared using the gcc intrinsics.

The library is compiled using a makefile. First try the following
commands to compile. If they don't work, edit the makefile to adjust
the configuration.

To compile the Matlab interface, do 

  make mex  

For the Octave (Maltab work-alike) interface: 

  make octave

For the Python interface: 

  make py 

Additional software and data
----------------------------

Although the library is self contained, the examples rely on:

- Yael (https://gforge.inria.fr/projects/yael/), and its Matlab
interface. The 

- H. Jegou's Matlab implementation of PQcodes
(http://www.irisa.fr/texmex/people/jegou/ann.php) for the PQ encoding. 

- some datafiles. See the JSGD homepage
(http://lear.inrialpes.fr/src/jsgd/). All examples are provided in a
simplistic format: a file containing the descriptor matrix + one with
the label vector, for train and test (4 files in total). The files are
in the .fvecs and .ivecs formats.

The examples assume that they are installed in the jsgd directory,
next to c/, matlab/ etc. On a normal Linux installation, the following
commands download and install all the required files (run from the
jsgd-xx directory):

# get Yael (need svn version...)
svn checkout --username anonsvn   --password anonsvn --no-auth-cache  https://scm.gforge.inria.fr/svn/yael/trunk yael

# compile
cd yael
./configure.sh --enable-numpy
make

# compile matlab interface 
cd matlab
make
cd ../..

# get pqcodes 
wget http://www.irisa.fr/texmex/people/jegou/src/pqsearch_2188.tar.gz
tar xvzf pqsearch_2188.tar.gz

# get additional data
wget http://pascal.inrialpes.fr/data2/douze/jsgd_data/groupFungus_k64_nclass134_nex50.tgz
tar xzf groupFungus_k64_nclass134_nex50.tgz


Interfaces
==========

There are two high-level interfaces to jsgd_train, a Matlab and a
Python interface. They provide a Matlab/Python jsgd_train function
that takes Matlab/Python matrices as input. The algorithm parameters
are passed as optional arguments to the functions.


Matlab interface
----------------

The matlab interface uses single matrices for floating-point data and
int32 matrices for indices, labels, etc. 

Python interface
----------------

The Python interface uses numpy matrices for i/o. The matrices should
be "c_compact", which means that all matrix dimensions in the
documentation should be transposed. The matrices should be of type
numpy.float32 or numpuy.int32 for int and float matrices
respectively. You should also set the PYTHONPATH to point on
jsgd-xx/yael, eg.

export PYTHONPATH=../yael

Examples
========

Examples are provided in Matlab and Python for readability. 

Matlab implementation
---------------------

A Matlab implementation of SGD is in the ref/ subdirectory. It is a
straightforward translation of the method described in the
article. The computations in the optimized version are based on this.

  sgd_simple.m:           SGD implementation

  generate_toy_data.m:    randomly generate a training and a test set.

  sgd_simple_toy_data.m:  runs both, computes the accuracy and shows 
                          results graphically.

Simple example scripts
----------------------

Please make sure the "additional software and data" (see above) is
installed. The scripts 

  python/test_jsgd.py 

  matlab/test_jsgd_train.m

show how to perform training on provided input sets, with fixed
parameters. See the scripts on how to use.

Product quantizer encoding 
--------------------------

The script 

 matlab/test_pq_encoded.m

shows how to learn a product quantizer, how to encode the training set
and how to train a classifier on the encoded training data.

Cross-validation of parameters
------------------------------

The SGD is only as good as the parameters it is computed with. The
default parameters will produce a suboptimal classifier. The main
parameters are:

  lambda             penalty term for W norm
  bias_term          this term is appended to each training vector
  eta0               intial value of the gradient step 
  beta               for OVR, how many negatives to sample per training vector

The script 

  python/crossval.py

Optimizes the parameters by stepping from a parameter set to another,
monitoring the corss-validated accuracy in the meantime. Unfortunately, 
this relatively exaustive exploration of the parameter space is slow 
(see below for a few tips on determining optimal parameters).

Comparison with supplementary material of the CVPR paper
--------------------------------------------------------

The script 

  python/example_BOW.py

reproduces the results from the supplementary material of the CVPR
paper (table 2, Ungulate with 50 examples per class). It shows how to
optimize a subet of parameters and to use sparse descriptors (it is
based on a BOW in 4096D). 

Library layout
==============

The main entry point for the library, both in C, Matlab and Python is 

  jsgd_train

There are a few mandatory parameters to jsgd_train, and many optional
ones. Dense matrices are stored contiguously in memory with a
column-major (Fortran/Matlab) convention, hence 

  x(d, n) 

means that matrix x has d lines and n columns.

Mandatory parameters
--------------------

The examples should already be shuffled randomly on input.

  x(d, n) is a floating-point matrix containing the n example
          vectors of dimension d. It can be encoded with a product
          quantizer if necessary (see below). 

  y(n)    is an interger array that contains the label associated with
          each column of x. Labels are 0-based in C and Python,
          1-based in Matlab.


Return values
-------------

The function returns 

  w(d + 1, nclass) a floating-point array such that 

                       max(w(:, i)' * [x ; 1])

                   gives the class of vector x

  stats            a structure (Matlab) or class (Python) that contain 
                   statistics on the resolution (timings, evaluations
                   on the validation set, etc.)


Optional parameters
-------------------

In Matlab, optional parameters are passed  

   'option_name', option_value 

pairs. In Python, use

   option_name = option_value

In C, just set the fields of the jsgd_params_t structure.

The most important ones: 

  lambda             penalty term for W norm (_lambda in Python)
  bias_term          this term is appended to each training vector
  eta0               intial value of the gradient step */
  beta               for OVR, how many negatives to sample per training vector
  fixed_eta          (boolean) use a fixed step

  verbose            verbosity level (integer, default 0)
  n_thread           if non-zero, try to use threads (for now)
  random_seed        running with the same random seed should give the same results
  algo               string, 'ovr', 'mul', 'rnk' or 'war' selecting the algorithm
  n_epoch            number of passes through the data
  
  valid              matrix of validation examples
  valid_labels       corresponding labels
  eval_freq          evaluate on validation set every that many epochs
  stop_valid_threshold 
                     stop if the accuracy is lower than the accuracy
                     at the previous evaluation + this much. Set to
                     something like -0.05 to stop iterating when all
                     hope is lost to obtain a better operating point.

Determining optimal parameters of SGD
=====================================

Here a few practical remarks on setting the parameters. 

The optimization routine implemented in crossval.py normally outputs
optimal parameters. Unfortunately, it is very slow. Below are a few
notes on how to choose reasonable parameters in finite time.

It is possible to fix some parameters and optimize on the others,
which speeds up the estimation. See example_BOW.py for an
example. 

An intermediate result of optimize() can be used (see the "keep"
messages of the cross-validation.

For OVR, beta is an important parameter. Setting it to the square root
of the number of classes seems a good starting point (if beta = 0 on
input, the algorithm sets this automatically).

L. Bottou's optimization of eta0 (choosing eta0 that minimizes the
loss on a subset) is implemented in jsgd_train. It will be used if the
passed-in eta0 is 0.

It is not useful to fine-tune the parameters very precisely. The
optimization criterion (top-1 accuracy on cross-validated datasets) is
quite noisy, so choosing small steps on the parameters leads to local
minima. Therefore, crossval.py optimizes lambda, eta0 and bias_term by
powers of 10.

If a validation set is available, do early stopping, and evaluate
every 10 or so epochs. For this, set stop_valid_threshold = -0.05,
which will abandon when the current accuracy is 5 point below the best
one seen so far.

The CVPR paper is also a good source on the influence of the various
parameters.

Low-level library
=================

The C library has one main function: 

int jsgd_train(int nclass,
               const x_matrix_t *x,
               const int *y,
               float * w, 
               float * bias, 
               jsgd_params_t *params); 

Where: 

 nclass        nb of classes
 x             matrix of train vectors
 y(n)          labels of each vector. Should be in 0:nclass-1
 w(d,nclass)   output w matrix
 bias(nclass)  output bias coefficients
 params        all other parameters 

A matrix a(m, n) of n float vectors of dimension m is represented as

  float *a

The (i, j)'th element of the matrix, where 0 <= i < m and 0 <= j < n
can be accessed with

  a[ j * m + i ]

x_matrix_t
----------

The matrix of training vectors is a x_matrix_t structure, because it
is not necessarily a dense matrix. Several encodings are used for x_matrix_t's:

** encoding =  JSGD_X_FULL

This is a plain dense matrix. The array 
   
  data(d, n) 

contains the elements.

** encoding = JSGD_X_PQ

This is a product quantizer matrix. There are nsq
subquantizers. Quantization centroids are given by the 3D matrix

  centroids(d / nsq, 256, nsq)

The corresponding codes are in array

  codes(nsq, n)

** encoding = JSGD_X_SPARSE

The matrix is a sparse column matrix (a la Matlab). html for the
layout.

  indptr(n + 1)    points to the the first non-0 cell in column j
  indices(nnz)     indices[indptr[j]] .. indices[indptr[j]] are 
                   the non-0 rows in column j (nnz = indptr[n])
  sparse_data(nnz) sparse_data[indptr[j]] .. sparse_data[indptr[j]] are 
                   the values of the cells in column j.

This is similar to the Matlab encoding and that of CSR in
scipy.sparse. See http://www.mathworks.fr/help/techdoc/apiref/mxsetir.


jsgd_params_t
-------------

The jsgd_params_t contains all the parameters of the training
algorithm, as well as output statistics produced when running it. See
above and c/jsgd.h for an explanation of its fields.

Optimizations
=============

The four algorithms can be expressed as (see ref/sgd_simple.m)

  for t = 1 to n 
  
    # draw a_set_of_classes more or less randomly
    for c in a_set_of_classes
       # a dot product
       score(t, c) := W(:, c)' * x(:, t)
    end for 
  
    # compute factor from the score table
  
    for c in a_subset_of_classes
       # a W update
       W(:, c) := scalar * W(:, c) + factor(c) * x(:, t)
    end for 
  
  end for

The operations that depend on d (the dimension of the vectors) are the
dot products and the W updates. Depending on the algorithm, each
iteration does

        Nb of dot products        Nb of W updates 
OVR     1 + beta                  1 + beta
MUL     C                         0 or 2 (1.08)
RNK     2                         0 or 2 (0.43)
WAR     2 to C (6)                0 or 2 (0.72)

where C is the number of classes, the numbers in parentheses are
measured on a run with C = 10 classes.

The library (especially the OVR version) is optimized in several ways:

- by using multipliers. In this case, w is represented as (w_factor *
w). Updates like 

  w := scalar * w 

are never actually performed, because only w_factor is updated. This
optimization is borrowed from Leon Bottou's reference implementation.

- by using SSE instructions. Operations are performed on 4 vectors
components at a time for dense and PQ-encoded matrices. If addresses
are not aligned on a 16-byte boundary or the vector size is not a
multiple of 4, the library falls back on the slow implementation. For
the caller, this means that if the vector dimensions are not a
multiple of 4, it is a good idea to pad them with 0s to the next
multiple of 4.

- for dense matrix-matrix and matrix-vector multiplications, BLAS is
used. Unfortunately, most operations cannot be expressed in these
terms (else the Matlab implementation would be fast enough). 

- the performance bottleneck that remains is cache accesses. Assuming
that there is 1MB of L3 cache per processor core, with 4096D single
precision descriptors, only 64 of them fit in cache. If
a_set_of_classes is all classes (like with MUL) and if C > 64, then
there will be cache misses at each iteration, which cost more than the
operations performed on the vectors. Therefore, for OVR, the code is
laid out in blocks like

  for t0 = 1 to n by t_step 
    for c = 1 to C

      for t = t_block to t_block + t_step - 1
        # compute all scores depending on t, c 
        # perform updates on W(:, c)
      end for 

    end for
  end for

where t_step is chosen so that all x(:, t) accessed in the inner loop
fit in cache.

A split in blocks around the d dimension is also implemented. It works
by precomputing

  self_dp(t1, t2) = x(:, t1)' * x(:, t2)

for all t0 <= t1, t2 < t0 + t_step. Then, at each t0 block start, we
compute

  score2(t, c) := W_t0(:, c)' * x(:, t)

  for all c and t0 <= t < t0 + t_step

where W_t0 means "the state of W at t = t0". This operation is easily
split around the dimension d. The score2 table is not the same as the
score table because it is computed with W's state at the start of the
t block. However W_t can be represented as

  W_t(:, c) = W_t0(:, c) + alpha_t0 * x(:, t0) + ... + alpha_{t-1} * x(:, t - 1)

Where the alpha_t's are computed at each W update. Therefore the
current score expands to

  score(t, c) = W_t(:, c)' * x(:, t)
              = score2(t, c) + alpha_t0 * self_dp(t0, t) + ... + alpha_{t-1} * self_dp(t, t-1)

which makes it possible to compute alpha_t. Computations of scores and
alpha's do not involve any operation in d. The actual update of W is
performed at the end of the t block using the operation above. 

This method is implemented (use_self_dotprods = 1 and set
d_step). Unfortunately, it is quite complicated, and even worse, it
does not seem to improve the speed, so it is disabled by default.

- when possible, multithreading is used. This is not easy when there
are dependencies between the factors and the scores across
classes. This is not the case for OVR: for OVR multithreading is
peformed along the classes (ie. each thread takes care of a subset of
classes).


Legal, contact
==============

This code is copyright (C) INRIA 2012

Homepage: http://lear.inrialpes.fr/src/jsgd/

License: Cecill (see http://www.cecill.info/licences/Licence_CeCILL_V1.1-US.txt), similar to GPL

Contact: Matthijs Douze, matthijs.douze@inria.fr

Last update 2012-06-07


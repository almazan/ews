from yael import ynumpy
import numpy

import os, pdb, sys

try:
  import scipy.sparse
except ImportError:  
  pass


def load_BOW(group, k, nclass, nex):
  """ Load BOW descriptors in dimension k, for at most nclass classes,
  keeping just nex training examples per class"""

  # basedir = '/scratch2/clear/paulin/datasets/%s/descs/dense_siftnonorm_bof%d/' % (group, k)
  basedir = '../example_data/%s_BOW%d/' % (group, k)

  # classes = nXXX numbers in alphabetical order
  nx = [f for f in os.listdir(basedir) if f.endswith('.fvecs')]
  nx.sort()

  # define train and test 
  Xtrain = []
  Xtest = []

  Ltrain = []
  Ltest = []
  
  for i, fname in enumerate(nx[:nclass]):

    print fname
    
    descs = ynumpy.fvecs_read(basedir + fname)
    assert descs.shape[1] == k 

    ni = descs.shape[0]

    # normalizations: power norm + L1 normalization
    descs = numpy.sqrt(descs)
    descs /= descs.sum(axis = 1).reshape(ni, 1)

    
    ntrain = ni / 2
    # keep only nex of the ntrain learning parameters
    Xtrain.append(descs[0:nex, :])
    Ltrain.append(i * numpy.ones((1, nex), dtype = numpy.int32))

    # the rest is for testing
    Xtest.append(descs[ntrain:ni, :])
    Ltest.append(i * numpy.ones((1, ni - ntrain), dtype = numpy.int32))

  # stack the matrices
  Xtrain = numpy.vstack(Xtrain)
  Ltrain = numpy.hstack(Ltrain)

  Xtest = numpy.vstack(Xtest)
  Ltest = numpy.hstack(Ltest)

  return Xtrain, Ltrain.T, Xtest, Ltest.T


# training examples per class
nex = 50
# limit to this many classes
nclass = 183

# load data
Xtrain, Ltrain, Xtest, Ltest = load_BOW('Ungulate', 4096, nclass, nex)

n, d = Xtrain.shape

print "train size %d vectors in %dD, %d classes " % (n, d, nclass)

# random permutation of training
numpy.random.seed(0)

perm = numpy.random.permutation(n)
Xtrain = Xtrain[perm, :]
Ltrain = Ltrain[perm,:]

# split test in valid + test
ntest = Ltest.size

perm = numpy.random.permutation(ntest)

# keep 5000 images for validation
nvalid = 5000
Xvalid = Xtest[perm[:nvalid], :]
Lvalid = Ltest[perm[:nvalid]]

# real test is the rest
Xtest = Xtest[perm[nvalid:], :]
Ltest = Ltest[perm[nvalid:]]
ntest = Xtest.shape[0]

print "test size: %d, valid size: %d" % (ntest, nvalid)

if True:
  # train with sparse matrices. A bit faster and same result.
  Xtrain = scipy.sparse.csr_matrix(Xtrain)
  Xvalid = scipy.sparse.csr_matrix(Xvalid)

if False: 

  # finds optimal parameters by cross validation
  from crossval import * 
  
  co = CrossvalOptimization(Xtrain, Ltrain)

  # change some of the default parameters
  co.max_epoch = 200

  # we do not optimize on eta0, so we fix it to 0 here (ie. use
  # Bottou's algorithm to find eta0 at the beginning of jsgd_train.
  co.constant_parameters['eta0'] =  0

  # starting point
  co.init_point = {'_lambda': 1e-06, 'beta': 20, 'bias_term': 0.01}

  # perform optimization (result on screen)
  co.optimize()

else:

  from jsgd import *
       
  W = jsgd_train(Xtrain, Ltrain,
                 valid = Xvalid,
                 valid_labels = Lvalid,
                 eval_freq = 5,
                 n_epoch = 500,
                 verbose = 2,                       
                 n_thread = 1,       # use threads
                 _lambda = 1e-7,
                 beta = 20,
                 eta0 = 0,
                 bias_term = 0.01,
                 stop_valid_threshold = -0.02)
                 
  # evaluate the found W
  Xtest1 = numpy.hstack([Xtest, numpy.ones((ntest, 1), dtype = numpy.float32)])

  # classification scores 
  scores = numpy.dot(W, Xtest1.T)
  
  # label = max score 
  found_labels = numpy.argmax(scores, axis = 0)
  
  # any more elegant way of expressing this welcome
  test_accuracy = sum([1 for i in (found_labels == Ltest.T)[0] if i]) / float(ntest)
  
  print "classification score on test: ", test_accuracy
  
  


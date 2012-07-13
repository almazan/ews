


from yael import ynumpy, yael
from jsgd import *


# mini-problem: 10 fungus classes, 10 examples / class, 4096D descriptors
# basename = "../example_data/groupFungus_k64_nclass10_nex10"

# medium: all fungus classes, 50 examples / class, 4096D descriptors
basename = "../example_data/groupFungus_k64_nclass134_nex50"

# 50 imagenet classes, 50 images / class (+50 for testing), 128 k-dimensional descriptors
# basename = 'data/imagenet_cache/k1024_nclass50_nex50'


# load training data
Xtrain = ynumpy.fvecs_read(basename + '_Xtrain.fvecs')
Ltrain = ynumpy.ivecs_read(basename + '_Ltrain.ivecs')
# shift to get 0-based labels
Ltrain = Ltrain - 1


# load test data
Xtest = ynumpy.fvecs_read(basename + '_Xtest.fvecs')
Ltest = ynumpy.ivecs_read(basename + '_Ltest.ivecs')
Ltest = Ltest - 1

# random permutation of train 
n = Xtrain.shape[0]
  
numpy.random.seed(0)
perm = numpy.random.permutation(n)
  
Xtrain = Xtrain[perm, :]
Ltrain = Ltrain[perm, :]
  
n_epoch = 6000
  
W, stats = jsgd_train(Xtrain, Ltrain,
                      valid = Xtest,
                      valid_labels = Ltest,
                      eval_freq = n_epoch,
                      n_epoch = n_epoch,
                      t_block = 4,
                      verbose = 2,                       
                      want_stats = True,
                      n_thread = 1)

print "final classification score = ", stats.valid_accuracies[-1]

# append a 1 at the end of each element
Xtest1 = numpy.hstack([Xtest, numpy.ones((n, 1), dtype = numpy.float32)])

# classification scores 
scores = numpy.dot(W, Xtest1.T)

# label = max score 
found_labels = numpy.argmax(scores, axis = 0)

# any more elegant way of expressing this welcome
test_accuracy = sum([1 for i in (found_labels == Ltest.T)[0] if i]) / float(n)

print "classification score computed in Python: ", test_accuracy

import pdb

import jsgd_wrap
import numpy



def _check_row_float32(a): 
  if a.dtype != numpy.float32: 
      raise TypeError('expected float32 matrix, got %s' % a.dtype)
  if not a.flags.c_contiguous:
      raise TypeError('expected C order matrix')


def _numpy_to_xmatrix(m):
  xm = jsgd_wrap.x_matrix_t()
  xm.n, xm.d = m.shape
  if not hasattr(m, 'indptr'): # dense
    _check_row_float32(m)  
    xm.encoding = jsgd_wrap.JSGD_X_FULL
    xm.data = jsgd_wrap.numpy_to_fvec_ref(m)
  else:     # sparse    
    if m.format != 'csr': 
      raise TypeError('expected sparse CSR matrix')
    xm.encoding = jsgd_wrap.JSGD_X_SPARSE
    xm.sparse_data = jsgd_wrap.numpy_to_fvec_ref(m.data)
    xm.indptr = jsgd_wrap.numpy_to_ivec_ref(m.indptr)
    xm.indices = jsgd_wrap.numpy_to_ivec_ref(m.indices)
  return xm
  
def _get_ptr(m):
  if m.dtype == numpy.int32: return jsgd_wrap.numpy_to_ivec_ref(m)
  if m.dtype == numpy.float32: return jsgd_wrap.numpy_to_fvec_ref(m)
  if m.dtype == numpy.float64: return jsgd_wrap.numpy_to_dvec_ref(m)
  raise TypeError('type %s not handled' % m.dtype)
  

class Stats:
  pass

def jsgd_train(x, y,
               valid = None,
               valid_labels = None,
               want_stats = False,
               algo = None,
               **kwargs):
  
  n, d = x.shape

  if y.size != n or y.dtype != numpy.int32 or not y.flags.c_contiguous:
    raise TypeError('labels have wrong size or type')
    
  params = jsgd_wrap.jsgd_params_t()
  jsgd_wrap.jsgd_params_set_default(params)
  
  nclass = int(y.max() + 1)

  assert nclass < 100000 and y.min() == 0, "weird labels"
 
  train_matrix = _numpy_to_xmatrix(x)
  
  if valid != None or valid_labels != None:
    assert valid != None and valid_labels != None
    valid_matrix = _numpy_to_xmatrix(valid)
    params.valid = valid_matrix
    params.valid_labels = _get_ptr(valid_labels)

  if algo != None:
    params.algo = getattr(jsgd_wrap, "JSGD_ALGO_" + algo.upper())

  # all other arguments are converted automagically
    
  for k, v in kwargs.items():
    assert hasattr(params, k)
    setattr(params, k, v)


  # output

  if want_stats:
    params.na_stat_tables = params.n_epoch / params.eval_freq + 1;
    stats = Stats()    
    fields = [
      ("valid_accuracies", numpy.float64), ("times", numpy.float64), ("train_accuracies", numpy.float64),
      ("ndotprods", numpy.int32), ("nmodifs", numpy.int32)];
    for field_name, np_type in fields:      
      a = numpy.zeros(params.na_stat_tables, dtype = np_type)
      setattr(stats, field_name, a)
      setattr(params, field_name, _get_ptr(a))
    

  
  W = numpy.zeros((nclass, d), dtype = numpy.float32)
  biases = numpy.zeros((nclass, 1), dtype = numpy.float32)  

  # the actual call

  jsgd_wrap.jsgd_train(nclass, train_matrix, _get_ptr(y),
                       _get_ptr(W), _get_ptr(biases),
                       params)

  # bias = additional line in W
  W = numpy.hstack([W, biases * params.bias_term])
  
  if want_stats:
    for field in  'niter', 'ndp', 'nmodif', 't_eval', 'best_epoch':
      setattr(stats, field, getattr(params, field))  
    return W, stats
  else:
    return W
  
  
  

  

if __name__ == '__main__':
  import sys
  
  from yael import ynumpy, yael

  print "loading..."

  # basename = 'data/fuv_cache/cache_groupFungus_k64_nclass132_nex50'
  # basename = 'data/BOW_florent/vehicle262_train_50ex'
  # basename = 'data/BOW_florent/vehicle262_train_50ex'
  # basename = 'data/BOW_florent/ungulate183_train_50ex'
  # basename = 'data/fuv_cache/cache_groupFungus_k64_nclass134_nex10000'
  # basename = 'data/fuv_cache/cache_groupFungus_k64_nclass132_nex50'
  # basename = 'data/fuv_cache/cache_groupFungus_k256_nclass134_nex10000'
  # basename = 'data/fuv_cache/cache_groupFungus_k256_nclass134_nex10000'
  # basename = 'data/fuv_cache/cache_groupVehicle_k256_nclass262_nex50'
  basename = 'data/imagenet_cache/k1024_nclass50_nex50'
  
  Xtrain = ynumpy.fvecs_read(basename + '_Xtrain.fvecs')
  Ltrain = ynumpy.ivecs_read(basename + '_Ltrain.ivecs')

  Ltrain = Ltrain - 1

  
  # basename = "data/BOW_florent/vehicle262_train_first50"
  # basename = "data/BOW_florent/ungulate183_train"
  
  Xvalid = ynumpy.fvecs_read(basename + '_Xtest.fvecs')
  Lvalid = ynumpy.ivecs_read(basename + '_Ltest.ivecs')
  Lvalid = Lvalid - 1

  # Xvalid = Xvalid[:10,:]
  # Lvalid = Lvalid[:10,:]

  n = Xtrain.shape[0]
  
  numpy.random.seed(0)
  perm = numpy.random.permutation(n)
  
  Xtrain = Xtrain[perm, :]
  Ltrain = Ltrain[perm, :]
  
  # params = {'_lambda': 0.0001, 'beta': 8, 'eta0': 0.1, 'bias_term': 0.5}
  # params = {'_lambda': 0.0001, 'beta': 0, 'eta0': 0, 'bias_term': 0.5} 
  # n_epoch = 60
  # n_epoch = 20


  beta = 4
  params = {'_lambda': 1e-7, 'beta': beta, 'eta0': 0.1, 'bias_term': 0.1}
  n_epoch = 2


  t_block = 16
  n_wstep = 32

  print params, t_block, n_wstep
    
  
  W,stats = jsgd_train(Xtrain, Ltrain,
                       valid = Xvalid,
                       valid_labels = Lvalid,
                       eval_freq = 2,
                       n_epoch = n_epoch,
                       verbose = 2,                       
                       want_stats = True,
                       n_thread = 1,
                       t_block = t_block,
                       use_self_dotprods = 1,
                       n_wstep = n_wstep,
                       **params)
  print "times:", stats.times
  print "time for eval:", stats.t_eval
  print stats.valid_accuracies
  
  
               
  

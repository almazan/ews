import time, sys, random, cPickle

from yael import ynumpy, threads, yael
from jsgd import *



########### A few small functions


  

def params_to_tuple(params):
  "convert param block to tuple (suitable as key in a hash table) " 
  if params == None: return None
  return tuple(sorted(params.items()))


class CrossvalOptimization:
  """
  Optimizes the parameters _lambda bias_term eta0 beta using
  cross-validation. First call the constructor, adjust optimization
  parameters and call optimize(), which returns the set of optimal
  parameters it could find.

  """

  def __init__(self, Xtrain, Ltrain):
  
    # maximum number of epochs we are willing to perform
    self.max_epoch = 100

    # frequency of evaluations
    self.eval_freq = 5
    
    # starting point for optimization (also specifies which parameters
    # should be considered for optimization)
    self.init_point = {
      '_lambda': 1e-4, 
      'bias_term': 0.1, 
      'eta0': 0.1, 
      'beta': 5}
    
    # additional parameters to jsgd_train
    self.constant_parameters = {
      'n_thread': 1,         # use threading
      'fixed_eta': False,    # constant eta?
      'algo': 'ovr',         # algorithm
      }

    # nb of cross-validation folds to use
    self.nfold = 5
    
    # consider all starting points that are within 5% of best seen so far 
    self.score_tol = 0.05
    
    # this tolerance decreases by this ratio at each iteration
    self.score_tol_factor = 0.5 

    self.Xtrain = Xtrain
    self.Ltrain = Ltrain

    self.nclass = max(self.Ltrain) + 1

    pow10range = [ 10**i for i in range(-8,8) ]

    # ranges for parameters in init_point
    self.ranges = {
      '_lambda': [0.0] + pow10range,
      'bias_term': pow10range,
      'eta0': pow10range,
      'beta': [b for b in [2,5,10,20,50,100,200,500] if b < self.nclass] }
    
    # all params can be changed after constructor
         

  def eval_params(self, params, fold):

    n = self.Xtrain.shape[0]

    # prepare fold
    i0 = fold * n / self.nfold
    i1 = (fold + 1) * n / self.nfold
    
    valid_i = numpy.arange(i0, i1)
    train_i = numpy.hstack([numpy.arange(0, i0), numpy.arange(i1, n)])

    ni = train_i.size
    kw = params.copy()
    for k, v in self.constant_parameters.items():
      kw[k] = v

    W,stats = jsgd_train(self.Xtrain[train_i, :],
                         self.Ltrain[train_i, :],
                         valid =        self.Xtrain[valid_i,:],
                         valid_labels = self.Ltrain[valid_i,:],
                         eval_freq = self.eval_freq,
                         n_epoch = self.max_epoch,                         
                         want_stats = True,
                         **kw)  
    return stats

  def params_step(self, name, pm):
    " step parameter no in direction pm "
    new_params = self.params.copy()
    if name == None and pm == 0: return new_params
    curval = self.params[name]
    r = self.ranges[name]
    i = r.index(curval)
    try: 
      new_params[name] = r[i + pm]
      return new_params
    except IndexError:
      return None


  def do_exp(self, (pname, pm, fold)):
    # perform experiment if it is not in cache
    params_i = self.params_step(pname, pm)

    k = (params_to_tuple(params_i), fold)
    if k in self.cache:
      res = self.cache[k]
    else: 
      res = self.eval_params(params_i, fold)
      self.cache[k] = res

    return (pname, pm, fold), res


    
  def optimize(self):

    self.nclass = int(max(self.Ltrain) + 1)
    nfold = self.nfold
    
    # cache for previous results
    self.cache = dict([((None,i),None) for i in range(self.nfold)])
    
    # best scrore so far
    best_score = 0

    t0 = time.time()
    it = 0

    queue = [(0, self.init_point)]

    while queue:

      # pop the best configuration from queue
      score_p, self.params = queue.pop()

      print "============ iteration %d (%.1f s): %s" % (
        it, time.time() - t0, self.params)
      print "baseline score %.3f, remaining queue %d (score > %.3f)"  % (
        score_p * 100, len(queue), (best_score - self.score_tol) * 100)

      # extend in all directions
      todo = [(pname, pm, fold)
              for pname in self.params
              for pm in -1, 1
              for fold in range(nfold)]

      if it == 0:
        # add the baseline, which has not been evaluated so far
        todo = [(None, 0, fold) for fold in range(nfold)] + todo

      # filter out configurations that have been visited already
      todo = [(pname, pm, fold) for (pname, pm, fold) in todo if
              (params_to_tuple(self.params_step(pname, pm)), 0) not in self.cache]

      # use multithreading if training itself is not threaded 
      if self.constant_parameters['n_thread']:
        n_thread = 1
      else:
        n_thread = yael.count_cpu()

      # perform all experiments 
      src = threads.ParallelIter(n_thread, todo, self.do_exp)  

      while True: 

        # pull a series of nfold results 
        try:   
          allstats = [ next(src) for j in range(nfold) ]
        except StopIteration:
          break

        (pname, pm, zero), stats = allstats[0]
        assert zero == 0
        params_i = self.params_step(pname, pm)

        # no stats for this point (may be invalid)
        if stats == None: continue

        params_key = params_to_tuple(params_i)

        # make a matrix of validation accuracies    
        valid_accuracies = numpy.vstack([stats.valid_accuracies for k, stats in allstats])

        # take max of average over epochs
        avg_scores = valid_accuracies.sum(0)
        score = avg_scores.max() / nfold
        sno = avg_scores.argmax()
        epoch = sno * self.eval_freq

        print "  %s, epoch %d, score = %.3f [%.3f, %.3f]" % (
          params_i, epoch,
          score * 100, 100 * valid_accuracies[:, sno].min(),
          100 * valid_accuracies[:, sno].max())

        if score >= best_score:
          # we found a better score!
          print "  keep"
          if score > best_score:
            best_op = []
            best_score = score
          best_op.append((params_i, epoch))

        # add this new point to queue
        queue.append((score, params_i))

        sys.stdout.flush()

      # strip too low scores from queue 
      queue = [(score, k) for score, k in queue if score > best_score - self.score_tol]

      # sorted by increasing scores (best one is last)
      queue.sort() 

      it += 1
      self.score_tol *= self.score_tol_factor


    print "best params found: score %.3f" % (best_score * 100)
    for params, epoch in best_op: 
      print params, epoch

    return [(params, epoch) for params, epoch in best_op]
    

if __name__ == '__main__': 
  # where to load the data from 
  basename = "../example_data/groupFungus_k64_nclass134_nex50"
   
  print "Loading train data %s" % basename

  Xtrain = ynumpy.fvecs_read(basename + '_Xtrain.fvecs')
  Ltrain = ynumpy.ivecs_read(basename + '_Ltrain.ivecs')
  
  # correct Matlab indices
  Ltrain = Ltrain - 1
  
  n, d = Xtrain.shape
  nclass = max(Ltrain) + 1

  print "train size %d vectors in %dD, %d classes " % (n, d, nclass)
  
  # random permutation of data
  
  numpy.random.seed(0)
  perm = numpy.random.permutation(n)
  
  Xtrain = Xtrain[perm, :]
  Ltrain = Ltrain[perm, :]

  co = CrossvalOptimization(Xtrain, Ltrain)

  # change defaults here if necessary

  co.optimize()



  



  



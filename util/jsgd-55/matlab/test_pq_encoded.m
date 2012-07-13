


addpath('../yael/matlab')
addpath('../pqcodes_matlab/')

rand('state', 0)

basename = '../example_data/groupFungus_k64_nclass134_nex50'

% load train data
Xtrain = fvecs_read([basename '_Xtrain.fvecs']);
Ltrain = ivecs_read([basename '_Ltrain.ivecs']);

% load test data
Xtest = fvecs_read([basename '_Xtest.fvecs']);
Ltest = ivecs_read([basename '_Ltest.ivecs']);




Xtrain = single(Xtrain); 

[d, n] = size(Xtrain);
perm = randperm(n); 

Xtrain = Xtrain(:, perm); 
Ltrain = Ltrain(perm);



% use 256 subquantizers
nsq = 256

% here we cache the result because it is slow to compute...

cachefname = sprintf('%s_pq%d.mat', basename, nsq);

if ~exist(cachefname, 'file') 
  
  fprintf(1, 'Learning a product quantizer on the training data\n');
  % train PQ data  
  pq = pq_new(nsq, Xtrain)
  
  fprintf(1, 'Storing PQ to %s\n', cachefname);
  save('-mat', cachefname, 'pq')
else 
  
  fprintf(1, 'Loading PQ from %s\n', cachefname);
  load(cachefname);
  
end
  
  

fprintf(1, 'encoding training set\n');
codes = pq_assign(pq, Xtrain);

n_epoch = 60; 

fprintf(1, '------------- Run without encoding (baseline result)\n');


[W,stats] = jsgd_train(Xtrain, int32(Ltrain), ...
                       'valid', single(Xtest), ...
                       'valid_labels', int32(Ltest), ...
                       'verbose', 2, ...
                       'eval_freq', n_epoch, ...
                       'n_epoch', n_epoch);

fprintf(1, '------------- Run with decoded PQ (loss due to PQ encoding)\n');

Xdecoded = pq_decode(pq, codes);

[W,stats] = jsgd_train(Xdecoded, int32(Ltrain), ...
                       'valid', single(Xtest), ...
                       'valid_labels', int32(Ltest), ...
                       'verbose', 2, ...
                       'eval_freq', n_epoch, ...
                       'n_epoch', n_epoch);

fprintf(1, '------------- Run with PQ encoded data (should give same result as above)\n');

% PQ data is passed as a struct in the function 
Xpq = struct();
Xpq.centroids = zeros(d / nsq, 256, nsq, 'single');
for q=1:nsq
  Xpq.centroids(:, :, q) = pq.centroids{q};
end
Xpq.codes = codes


[W,stats] = jsgd_train(Xpq, int32(Ltrain), ...
                       'valid', single(Xtest), ...
                       'valid_labels', int32(Ltest), ...
                       'verbose', 2, ...
                       'eval_freq', n_epoch, ...
                       'n_epoch', n_epoch);


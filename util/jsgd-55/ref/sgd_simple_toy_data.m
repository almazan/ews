%
% test the Matlab SGD on toy data 
%

rand('state', 0)

% dimension of data. 2 is nice for visualization, but typically too low for the data to be linearly separable. 
d = 2;        

nclass = 5;  % nb of classes 
nex = 200;   % nb of examples per class (train + test) 

[Xtrain, Ltrain, Xtest, Ltest] = generate_toy_data(d, nclass, nex, 0.05);

n = size(Xtrain, 2);
ntest = size(Xtest, 2);

% Graphic output
close all 
plot_with_labels(Xtrain, Ltrain)
title('Train data')

% training parameters
opt = struct(); 
opt.eval_freq = n;  % evaluate on validation set at each epoch

opt.otype = 'ovr';
opt.lambda = 1e-4;
opt.bias_term = 0.1;
opt.eta0 = 0.1;
opt.npass = 50 * n; % 50 epochs
opt.beta = 4;

if 0

  opt.otype = 'mul';
  opt.lambda = 1e-5;
  opt.bias_term = 1;
  opt.eta0 = 0.01;
  opt.npass = 50 * n;
  
  opt.otype = 'rnk';
  opt.lambda = 1e-5;
  opt.bias_term = 0.1;
  opt.eta0 = 0.01;
  opt.npass = 200 * n;

  opt.otype = 'war';
  opt.lambda = 1e-6;
  opt.bias_term = 1;
  opt.eta0 = 0.01;
  opt.npass = 100 * n;

end


% keep some train data for validation
nvalid = n / 5

opt.Xvalid = Xtrain(:, 1:nvalid);
opt.Lvalid = Ltrain(1:nvalid);

Xtrain = Xtrain(:,nvalid+1:end); 
Ltrain = Ltrain(nvalid+1:end);

% run SGD

W = sgd_simple(Xtrain, Ltrain, opt)

% evaluate W on test
[scores, found_labels]  = max(W' * [Xtest ; ones(1, ntest)]);
accuracy_on_test = sum(found_labels == Ltest) / ntest

% graphic output,  
figure
plot_with_labels(Xtest, found_labels); 
title('Classified test')

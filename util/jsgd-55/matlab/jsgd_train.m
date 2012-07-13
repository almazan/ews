% mex interface to the SGD linear classifier training function jsgd.
%
% [W, stats] = jsgd_train(X, y, ...)
%
% input:  
%   X: d-by-n matrix of training examples (single precision)
%   y: n vector of training labels (int32)
%
% output: 
%   W: (d + 1)-by-nclass classification matrix (+1 for the bias term)
%   stats: structure with statistics on the learning.
% 
% other parameters are passed in as options, as 
%
%   'option_name', option_value 
%
% pairs. They include:
% 
%   algo:      'ovr', 'mul', 'rnk', 'war' = variation on the training algorithm
%   lambda:    weighting of the regularization term
%   n_epoch:   number of passes to perform over the data
%   

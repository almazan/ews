% This function computes the fisher vector representation of a set of vectors
% See "Fisher kernels on visual vocabularies for image categorization"
%     by F. Perronnin and C. Dance, CVPR'2007
% 
% Usage: 
%   fishervector = yael_kmeans (v, w, mu, sigma)
%   fishervector = yael_kmeans (v, w, mu, sigma, 'opt1', 'opt2', ...)
%
% where
%   v is the set of descriptors to describe by the Fisher Kernel representation
%   w, mu and sigma are the parameters of the mixture (learned by, e.g., yael_gmm)
%
% 
% By default, only the components associated with variance mu are compuetd
%
% Options:
%   'weights'   includes the mixture weights in the representation
%   'sigma'     includes the terms associated with variacne
%   'nomu'      do not compute the terms associated with mean
%   'nonorm'    do not normalize the fisher vector
%   'verbose'   

%
% This script is a straightforward Matlab implementation of the SGD routines described in Perronnin CVPR 2012. 
%
% Xtrain: training examples (should be shuffled randomly on input)
% Ltrain: training labels
% opt: options structure

function W = sgd_simple(Xtrain, Ltrain, opt)

[d, n] = size(Xtrain)
nclass = max(Ltrain)

if isfield(opt, 'Xvalid')
  Xvalid = opt.Xvalid;
  Lvalid = opt.Lvalid;
else
  Xvalid = [];
  Lvalid = [];
end

Xtrain = [Xtrain ; ones(1, n) * opt.bias_term];
Xvalid = [Xvalid ; ones(1, size(Xvalid, 2)) * opt.bias_term];

if strcmp(opt.otype, 'war')
  lk = cumsum(1 ./ (1:nclass));
end

d_1 = d + 1;

W = zeros(d_1, nclass);

% number of W modifications 
nmodif = 0;

% number of dot products
ndp = 0;

tic;

best_accuracy_on_valid = -1;

for t = 1:opt.npass
  
  i = mod(t - 1, n) + 1;
  
  xi = Xtrain(:, i);
  yi = Ltrain(i); 

  eta = opt.eta0 / (1 + opt.lambda * opt.eta0 * t);
  fw = 1 - eta * opt.lambda;
  
  switch opt.otype
   
   case 'ovr'
    
    % choose w's to modify
    
    ybars = randperm(nclass);
    ybars(find(ybars == yi)) = ybars(1);  
    ybars = [yi ybars(2:opt.beta + 1)];   
    
    for ybar = ybars
      pm = (ybar == yi) * 2 - 1; % +/-1 label
      L_ovr = max(0, 1 - pm * W(:, ybar)' * xi);          

      ndp = ndp + 1;
      
      if L_ovr > 0
        % there is something to correct
        W(:, ybar) = W(:, ybar) * fw + eta * pm * xi;
        nmodif = nmodif + 1; 
      else 
        W(:, ybar) = W(:, ybar) * fw;
      end    
      
    end   
    
   case 'mul'
      
    scores = W' * xi + 1;    
    scores(yi) = scores(yi) - 1;
   
    ndp = ndp + nclass;
    
    % worst violation
    [max_val, ybar] = max(scores); 
    
    W = fw * W;
    
    if ybar ~= yi
      W(:, yi) = W(:, yi) + eta * xi;
      W(:, ybar) = W(:, ybar) - eta * xi;
      nmodif = nmodif + 2;  
    end      

   case 'rnk'
    
    % draw label ybar ~= yi
    ybar = 1 + floor(rand() * (nclass - 1)); 
    if ybar >= yi
      ybar = ybar + 1;
    end
    
    L_tri = max(0, 1 - W(:, yi)' * xi + W(:, ybar)' * xi); 
    ndp = ndp + 2; 
    
    W = fw * W;
    
    if L_tri > 0 
      % enforce at least 1 unit margin
      %  W(:, ybar) * xi  > W(:, yi) * xi - 1
      W(:, yi) = W(:, yi) + eta * xi;
      W(:, ybar) = W(:, ybar) - eta * xi;
      nmodif = nmodif + 2;        
    end
    
   case 'war'
    
    ybars = randperm(nclass);
    ybars(find(ybars == yi)) = [];

    score_yi = W(:, yi)' * xi;
    ndp = ndp + 1;
    
    
    for k = 1:nclass - 1
      ybar = ybars(k);
      L_tri = max(0, 1 - score_yi + W(:, ybar)' * xi); 
      ndp = ndp + 1;      
      if L_tri > 0
        break
      end      
    end
    
    W = fw * W;
    
    if L_tri > 0             
      W(:, yi) = W(:, yi) + eta * lk(k) * xi;
      W(:, ybar) = W(:, ybar) - eta * lk(k) * xi;
      nmodif = nmodif + 2;        
    end
      
    
   otherwise 
    error ('invalid optimization type');    
  end

  
  if mod(t, opt.eval_freq) == 0
    fprintf(1, 'Evaluation at pass %d (%.3f s), %d dot prods, %d W modifications\n',...
            t, toc(), ndp, nmodif);
    
    scores = W' * Xtrain; 
    [ ignored, found_labels ] = max(scores);
    accuracy_on_train = sum(found_labels == Ltrain) / length(Ltrain)
    if length(Xvalid) > 0
      scores = W' * Xvalid; 
      [ ignored, found_labels ] = max(scores);
      accuracy_on_valid = sum(found_labels == Lvalid) / length(Lvalid)    
      if accuracy_on_valid > best_accuracy_on_valid
        best_accuracy_on_valid = accuracy_on_valid;
        bestW = W;
      end
    end
  end

  
end
  
if length(Xvalid) > 0
  W = bestW;
end

% output W will be used with last component of vector set to 1
W(end, :) = W(end, :) * opt.bias_term; 

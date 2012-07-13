
function [X, labels] = read_svmlight(fname);
    
  f = fopen(fname, 'r');
  
  if f == -1 
    error(['cannot open ' fname ' for input'])
  end

  is = []; 
  js = []; 
  vals = [];
  labels = []; 
  i = 1; 
  
  while 1
    l = fgets(f); 
    if length(l) == 1 & l == -1
      break
    end
    
    [label, rest ] = strtok(l, ' ');
    labels = [ labels str2num(label) ]; 
    idx_val = sscanf(rest, '%d:%f '); 
    nnz = length(idx_val) / 2; 
    idx_val = reshape(idx_val, 2, nnz);
    is = [is (i * ones(1, nnz)) ]; 
    js = [js idx_val(1, :)]; 
    vals = [vals idx_val(2, :)];   
    i = i + 1; 
  end

  X = full(sparse(js, is, vals));
      
  fclose(f);

  

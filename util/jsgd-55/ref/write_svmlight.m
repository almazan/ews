


function write_svmlight(fname, train, labels);
    
  f = fopen(fname, 'w');
  
  if f == -1 
    error(['cannot open ' fname ' for output'])
  end
  
  assert(size(train, 2) == size(labels, 2)); 
  n = size(train, 2); 
  assert(size(labels, 1) == 1);
  d = size(train, 1); 
  
  for i = 1:n
    fprintf(f, '%d ', labels(i));
    
    if 0
    
      for j = 1:d
        if train(j, i) ~= 0
          fprintf(f, '%d:%-15.10g ', j, train(j, i));
        end
      end
    
    else      
      js = find(train(:, i) ~= 0); 
      fprintf(f, '%d:%-15.10g ', [js' ; train(js, i)']);            
    end
     
    
    fprintf(f, '\n');    
  end  
  
  fclose(f);

  

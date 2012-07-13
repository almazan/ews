% draw a set of 2D points with different colors for each label

function col = plot_with_labels(x, labels)
  colormap = 'bgrcmy';
  dots = '.o+*';
  l = length(colormap);  

  nclass = max(labels);
  
  for i = 1:nclass
    subset = find(labels == i);
    col = [colormap(mod(i - 1, l) + 1) dots(floor((i - 1) / l) + 1)];
    plot(x(1, subset), x(2, subset), col);
    hold on
  end    
    
end

function save_result_images(q, idq, locWords, resultLabels, docs, params)

path = sprintf('%s%d/',params.pathResultsImages, idq);
if ~exist(path, 'dir');
   mkdir(path);
end

imDocQ = imread(q.pathIm);
imq = imDocQ(q.loc(3):q.loc(4),q.loc(1):q.loc(2));
file = sprintf('000q.png');
imwrite(imq, [path file], 'png');

numIm = min(length(resultLabels), params.numResultImages);
for i = 1:numIm
    if resultLabels(i)==1
        flag = 'c';
    else
        flag = 'e';
    end    
    file = sprintf('%.3d%s.png', i, flag);
    
    bb = locWords(i,:);
    bb(3:4) = bb(3:4)-docs(bb(5)).yIni;
    bb(1:2) = bb(1:2)-docs(bb(5)).xIni;
    im = docs(bb(5)).image(bb(3):bb(4),bb(1):bb(2));
    imwrite(im, [path file], 'png');
end

end


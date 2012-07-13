function [scores, results, locW] = eval_model(params, model, docs, relevantBoxes, PQ_centroids)

numDocs = params.numDocs;

locWindows = cell(numDocs,1);
scoresWindows = cell(numDocs,1);
resultsWindows = cell(numDocs,1);

for i = 1:numDocs
    [scsW, locW] = get_windows(params, docs(i), model, PQ_centroids);
    
    
    % Non-maxima supression
    [t,I] = sort(scsW);
    I = I(end-params.thrWindows:end);
    pick = nms_C(int32(I),int32(locW)',params.overlapnms);
    
    scsW = scsW(pick);
    locW = locW(pick,:);
    locW = [locW repmat(i,length(scsW),1)];
    res = zeros(length(scsW),1);
    if ~isempty(relevantBoxes{i})
    
        relBoxes = relevantBoxes{i};
    
        relBoxes = [relBoxes(:,1) relBoxes(:,3) relBoxes(:,2)-relBoxes(:,1)+1 relBoxes(:,4)-relBoxes(:,3)+1];
        predBoxes = [locW(:,1) locW(:,3) locW(:,2)-locW(:,1)+1 locW(:,4)-locW(:,3)+1];
    
        intArea = rectint(single(predBoxes), single(relBoxes));
    
        areaP = model.newH * model.newW;
        areaGt = relBoxes(:,3).*relBoxes(:,4);
    
        denom = bsxfun(@minus, single(areaP+areaGt'), intArea);
        overlap = intArea./denom;
    
        [y,x] = find(overlap >= params.overlap);
        [U, PosY, PosX] = unique(x,'first');
        res(y(PosY))=1;
    end
    
    resultsWindows{i} = res;
    scoresWindows{i} = scsW;
    locWindows{i} = locW;
    
    % sacar interseccion (cuidado formato [x y w h])!!!
    % sacar union
end

%% Sort all the scores
scores = [];
results = [];
locW = [];
for i=1:numDocs
    scores = [scores; scoresWindows{i}];
    results = [results; resultsWindows{i}];
    locW = [locW; locWindows{i}];
end

[scores ,idx] = sort(scores,'descend');
results = results(idx);
locW = locW(idx,:);

% Supress the last non-relevant windows (does not affect to mAP)
index = find(results>0,1,'last');
scores = scores(1:index);
results = results(1:index);
locW = locW(1:index,:);
end

function [scsW, locW] = get_windows(params, doc, model, PQ_centroids)
% Extracts all the possible patches from a document images and computes the
% score given the queried word model

featDoc = doc.features;
bH = doc.bH;
bW = doc.bW;
xIni = doc.xIni;
yIni = doc.yIni;
nbinsH = model.bH; nbinsW = model.bW;
dim = size(model.root,1)/(nbinsH*nbinsW);
rangeH = 1:params.step:bH-nbinsH+1;
rangeW = 1:params.step:bW-nbinsW+1;
numWindows = length(rangeW)*length(rangeH);

if params.DoPQ
    % distances lookup
    dist_lookup = single(reshape(model.root, dim, nbinsH*nbinsW)'*PQ_centroids)';
    squares = single(sum(PQ_centroids.*PQ_centroids));
    scsW = compute_scores_PQ_L2(featDoc-1,dist_lookup, squares, bH,bW,nbinsH, nbinsW, params.step, numWindows);
else
    flat = single((featDoc(:)));
    scsW = compute_scores_L2(flat,single(model.root), bH,bW,dim,nbinsH, nbinsW, params.step, numWindows);
end

YY = (rangeH-1)*params.sbin+yIni;
XX = (rangeW-1)*params.sbin+xIni;
[q,p] = meshgrid(YY,XX);
locW=[p(:) p(:)+model.newW-1 q(:) q(:)+model.newH-1];

end
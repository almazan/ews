function [scsW, locW] = get_windows(params, doc, model, PQ_centroids)
% Extracts all the possible patches from a document images and computes the
% score given the queried word model
scsW = [];
locW = single([]);
for i = 1:params.numscales
    scale = params.scales(i);
    featDoc = doc.features{i};
    bH = doc.bH(i);
    bW = doc.bW(i);
    % xIni = doc.xIni;
    % yIni = doc.yIni;
    nbinsH = model.bH; nbinsW = model.bW;
    dim = size(model.root,1)/(nbinsH*nbinsW);
    rangeH = 1:params.step:bH-nbinsH+1;
    rangeW = 1:params.step:bW-nbinsW+1;
    numWindows = length(rangeW)*length(rangeH);
    
    if params.DoPQ
        % distances lookup
        dist_lookup = single(reshape(model.root, dim, nbinsH*nbinsW)'*PQ_centroids)';
        squares = single(sum(PQ_centroids.*PQ_centroids));
        scsW2 = compute_scores_PQ_L2(featDoc-1,dist_lookup, squares, bH,bW,nbinsH, nbinsW, params.step, numWindows);
    else
        flat = single((featDoc(:)));
        scsW2 = compute_scores_L2(flat,single(model.root), bH,bW,dim,nbinsH, nbinsW, params.step, numWindows);
    end
    scsW = [scsW; scsW2];
    
    YY = (single(rangeH)-1)*params.sbin*scale+1;
    XX = (single(rangeW)-1)*params.sbin*scale+1;
    [q,p] = meshgrid(YY,XX);
    locW=[locW; p(:) p(:)+single(model.newW-1) q(:) q(:)+single(model.newH-1)]; 
end
end
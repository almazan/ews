function [docs, PCA, PQ_centroids] = compute_features_docs(params, docs)
%% Computes the features of the input docs


if ~exist(params.fileFeatures, 'file')
    docs = get_images_docs(params, docs);
    for d=1:length(docs)
        % Computing features and labels (at bin-level) for the document image
        fprintf('Computing features and labels of doc %d\n', d);
        
        %% Extract features
        [bH,bW,tr1,docs(d).features] = esvm_features(docs(d).image,params.sbin,params.descrType);
        
        docs(d).bH = bH;
        docs(d).bW = bW;
    end
    
    %% Learn PCA
    PCAHOGS=[];
    hogsPerDoc = floor(params.NPCA / length(docs));
    for d=1:length(docs)
        rp = randperm(size(docs(d).features,2));
        z = docs(d).features;
        PCAHOGS = [PCAHOGS z(:,rp(1:hogsPerDoc))];
    end
    PCA.means = mean(PCAHOGS,2);
    centered = bsxfun(@minus, PCAHOGS, PCA.means);
    [V,D] = eig(cov(centered'));
    [PCA.eigvalues,idx]=sort(diag(D), 'descend');
    PCA.eigvectors = V(:,idx);
    
    if params.PCADIM > 0
        for d=1:length(docs)
            z = docs(d).features;
            centered = bsxfun(@minus, z, PCA.means);
            docs(d).features = PCA.eigvectors(:,1:params.PCADIM)'*centered;
        end
    end
    
    %% Learn PQ
    PQHOGS=[];
    hogsPerDoc = floor(params.NPCA / length(docs));
    for d=1:length(docs)
        rp = randperm(size(docs(d).features,2));
        z = docs(d).features;
        PQHOGS = [PQHOGS z(:,rp(1:hogsPerDoc))];
    end
    PQ_centroids = yael_kmeans(PQHOGS,256, 'redo',3,'niter',100);
    if params.DoPQ
        for d=1:length(docs)
            [idx,dis] = yael_nn(PQ_centroids, docs(d).features, 1, 2);
            docs(d).features = idx;
        end
    end
    
    
    %% Save
    save(params.fileFeatures, 'docs', 'PCA','PQ_centroids','-v7.3');
else
    load(params.fileFeatures);
end

end
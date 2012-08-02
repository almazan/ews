function [docs, PCA, PQ_centroids] = compute_features_docs(params, docs)
%% Computes the features of the input docs


if ~exist(params.fileFeatures, 'file')
    docs = get_images_docs(docs);
    %% Compute features for each document image
    for d=1:length(docs)
        
        fprintf('Computing features of doc %d\n', d);
        
        num_scales = length(params.scales);
        docs(d).Hscales = zeros(num_scales,1);
        docs(d).Wscales = zeros(num_scales,1);
        docs(d).bH = zeros(num_scales,1);
        docs(d).bW = zeros(num_scales,1);
        docs(d).features = cell(num_scales,1);
        
        % Compute pyramid of features
        for i = 1:num_scales
            im = imresize(docs(d).image, params.scales(i));
            newH = docs(d).H;
            newW = docs(d).W;
            res = mod(newH,params.sbin);
            while res~=0
                newH = newH-res;
                res = mod(newH,params.sbin);
            end
            res = mod(newW,params.sbin);
            while res~=0
                newW = newW-res;
                res = mod(newW,params.sbin);
            end
            difH = docs(d).H-newH;
            difW = docs(d).W-newW;
            im = im(1:end-difH,1:end-difW);
            
            docs(d).Hscales(i) = newH;
            docs(d).Wscales(i) = newW;
            
            % Extract features
            [docs(d).bH(i),docs(d).bW(i),tr1,docs(d).features{i}] = esvm_features(im,params.sbin,params.descrType);
        end
    end
    
    %% Learn PCA
    PCAHOGS=[];
    hogsPerDoc = floor(params.NPCA / length(docs));
    for d=1:length(docs)
        z = docs(d).features{params.orig_scale_index};
        rp = randperm(size(z,2));
        PCAHOGS = [PCAHOGS z(:,rp(1:hogsPerDoc))];
    end
    PCA.means = mean(PCAHOGS,2);
    centered = bsxfun(@minus, PCAHOGS, PCA.means);
    [V,D] = eig(cov(centered'));
    [PCA.eigvalues,idx]=sort(diag(D), 'descend');
    PCA.eigvectors = V(:,idx);
    
    if params.PCADIM > 0
        for d=1:length(docs)
            for i=1:num_scales
                z = docs(d).features{i};
                centered = bsxfun(@minus, z, PCA.means);
                docs(d).features{i} = PCA.eigvectors(:,1:params.PCADIM)'*centered;
            end
        end
    end
    
    %% Learn PQ
    PQHOGS=[];
    hogsPerDoc = floor(params.NPCA / length(docs));
    for d=1:length(docs)
        z = docs(d).features{params.orig_scale_index};
        rp = randperm(size(z,2));
        PQHOGS = [PQHOGS z(:,rp(1:hogsPerDoc))];
    end
    PQ_centroids = yael_kmeans(PQHOGS,256, 'redo',3,'niter',100);
    if params.DoPQ
        for d=1:length(docs)
            for i=1:num_scales
                [idx,dis] = yael_nn(PQ_centroids, docs(d).features{i}, 1, 2);
                docs(d).features{i} = idx;
            end
        end
    end
    
    
    %% Save
    save(params.fileFeatures, 'docs', 'PCA','PQ_centroids','-v7.3');
else
    load(params.fileFeatures);
end

end
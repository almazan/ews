function [docs, PCA, PQ_centroids] = compute_features_docs(params, docs)

if ~exist(params.fileFeatures, 'file')
    docs = get_images_docs(params, docs);
    for d=1:length(docs)
        % Computing features and labels (at bin-level) for the document image
        fprintf('Computing features and labels of doc %d\n', d);
        
        %% Extract features
        [bH,bW,tr1,docs(d).features] = esvm_features(docs(d).image,params.sbin,params.descrType);
        
        docs(d).bH = bH;
        docs(d).bW = bW;
        
        %% Compute labels at HOG level
        labelsW = zeros(bH,bW);
        labelsC = zeros(bH,bW);
        sbin = params.sbin;
        abin = sbin*sbin;
        words = docs(d).words;
        yIni = docs(d).yIni;
        xIni = docs(d).xIni;
        
        WordsPos = zeros(length(words), 4);
        for w=1:length(words)
            WordsPos(w,:)=[words(w).loc(1) words(w).loc(3) words(w).W words(w).H];
        end
        
%         iia = yIni+sbin*(1:bH);
%         jja = xIni+sbin*(1:bW);
%         iib = (1:bH);
%         jjb = (1:bW);
%         [iiia,jjja]=meshgrid(iia,jja);
%         [iiib,jjjb]=meshgrid(iib,jjb);
%         WindowsPos=[jjja(:),iiia(:), repmat(sbin, bH*bW,1), repmat(sbin, bH*bW,1), jjjb(:),iiib(:)];
%         intArea = rectint(WindowsPos(:,1:4), WordsPos);
%         [pv,pw] = find((intArea/abin) > 0.75);
%         ijw = [WindowsPos(pv, 6) WindowsPos(pv, 5) pw];
%         for ind=1:length(ijw)
%             trip = ijw(ind,:);
%             labelsW(trip(1),trip(2)) = int32(words(trip(3)).globalIdx);
%             labelsC(trip(1),trip(2)) = int32(words(trip(3)).class);
%         end
%         
%         docs(d).labelsHOGW = labelsW;
%         docs(d).labelsHOGC = labelsC;
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
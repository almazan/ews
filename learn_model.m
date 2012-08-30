function model = learn_model(params, query, docs, PCA, PQ_centroids)
% Learns the query model

rand('seed',0);rand('state',0);

model = struct;
pxbin = params.sbin;
% We expand the query to capture some context
loc = [query.loc(1)-pxbin query.loc(2)+pxbin query.loc(3)-pxbin query.loc(4)+pxbin];
newW = loc(2)-loc(1)+1; newH = loc(4)-loc(3)+1;
res = mod(newH,pxbin);
while res~=0
    newH = newH-res;
    loc(3) = loc(3)+floor(double(res)/2);
    loc(4) = loc(4)-ceil(double(res)/2);
    res = mod(newH,pxbin);
end
res = mod(newW,pxbin);
while res~=0
    newW = newW-res;
    loc(1) = loc(1)+floor(double(res)/2);
    loc(2) = loc(2)-ceil(double(res)/2);
    res = mod(newW,pxbin);
end
model.newH = newH;
model.newW = newW;
model.bH = newH/params.sbin-2;
model.bW = newW/params.sbin-2;
descsz = model.bH*model.bW*params.dim;

trHOGs = zeros(descsz,params.numTrWords+params.numNWords);

imDoc = imread(query.pathIm);
[H,W] = size(imDoc);

%% Get positive windows
ps = 1;
for dx=params.rangeX
    for dy=params.rangeY
        % Extract image patch
        x1 = max(loc(1)+dx,1); x2 = min(loc(2)+dx,W);
        y1 = max(loc(3)+dy,1); y2 = min(loc(4)+dy,H);
        im = imDoc(y1:y2,x1:x2);
        [h2,w2] = size(im);
        if h2~=newH || w2 ~=newW
            im = imresize(im,[newH,newW]);
        end
        
        % Extract descriptor. rows*cols X descsz.
        % Flattened by rows.
        [tr1,tr2,tr3,desc] = esvm_features(im,params.sbin,params.descrType);
        
        if dx==0 && dy==0
            im2 = repmat(im(:,:,1),[1 1 3]);
            descHOG = features_pedro(double(im2),params.sbin);
        end
        
        % Apply PCA
        if params.PCADIM > 0
            % Center
            desc = bsxfun(@minus, desc, PCA.means);
            desc = PCA.eigvectors(:,1:params.PCADIM)'*desc;
        end
        trHOGs(:,ps) = desc(:);
        ps = ps+1;
    end
end

%% Get negative windows
wordsByDoc = params.numNWords/params.numDocs;
startPos = params.numTrWords;
for id = 1:length(docs)
    %     scale = ceil(params.numscales/2);
    %     fD = docs(id).features{scale};
    %     BH = docs(id).bH(scale);
    %     BW = docs(id).bW(scale);
    
    numBins = model.bH*model.bW;
    %     IND = int32(zeros(1,wordsByDoc*numBins));
    IND = cell(params.numscales,1);
    for jj=1:wordsByDoc
        % Pick a random scale
        sc = max(round(params.numscales*rand),1);
        
        BH = docs(id).bH(sc);
        BW = docs(id).bW(sc);
        
        % Pick a random starting cell
        by = 1+round((BH-model.bH-1)*rand);
        bx = 1+round((BW-model.bW-1)*rand);
        % Get all the cell indices of the window, row-major.
        kk = 0;
        for tmpby=by:by+model.bH-1
            sp = (tmpby-1)*BW+bx;
            %             IND((jj-1)*numBins+kk*model.bW+1:(jj-1)*numBins+(kk+1)*model.bW) = sp:sp+model.bW-1;
            IND{sc} = [IND{sc} sp:sp+model.bW-1];
            kk = kk+1;
        end
    end
    if params.DoPQ
%                 rec=PQ_centroids(:,fD(IND));
        %%%%
        rec = [];
        for i=1:params.numscales
            rec = [rec PQ_centroids(:,docs(id).features{i}(IND{i}))];
        end
        %%%%
        descs = reshape(rec, descsz, wordsByDoc);
    else
        %%%%
        rec = [];
        for i=1:params.numscales
            rec = [rec; docs(id).features{i}(:,IND{i})];
        end
        %%%%
        %         descs = reshape(fD(:,IND), descsz, wordsByDoc);
        
    end
    trHOGs(:,startPos + (id-1)*wordsByDoc+1: startPos + (id)*wordsByDoc) = descs;
end

% Apply L2-norm
trHOGs=bsxfun(@times, trHOGs, 1./sqrt(sum(trHOGs.^2)));

%% Learn model
if strcmp(params.svmlib, 'jsgd')
    labels = ones(params.numTrWords+params.numNWords,1)*2;
    labels(1:params.numTrWords) = 1;
    p = randperm(size(trHOGs,2));
    trHOGs = trHOGs(:,p);
    labels = labels(p);
    model.root = jsgd_train(single(trHOGs), int32(labels), 'algo', 'ovr', ...
        'lambda', params.lambda,...
        'bias_term', params.bias,...
        'eta0', params.eta, ...
        'verbose', 2, ...
        'eval_freq', params.epochs, ...
        'n_epoch', params.epochs);
    model.root = model.root(1:end-1,1);
elseif strcmp(params.svmlib, 'bl')
    model.root = single(trHOGs(:,floor(size(params.rangeX,2)*size(params.rangeY,2)/2)));
end


end
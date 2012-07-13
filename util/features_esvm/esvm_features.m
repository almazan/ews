function [bH,bW,dim,feat] = esvm_features(I, sbin, descrType)
%Return the current feature function

if nargin == 0
    x = 31;
    return
end

if nargin == 2
    descrType = 'hog_esvm';
end

if size(I,3)~=3
    I = repmat(I(:,:,1),[1 1 3]);
end

if strcmp(descrType,'hog_esvm')
    x = features_pedro(double(I),sbin);
elseif strcmp(descrType,'hog_esvm13')
    x = features_pedro13(double(I),sbin);
end

[bH,bW,dim] = size(x);
feat = zeros(dim,bH*bW);
for h = 1:bH
    off = (h-1)*bW;
    for w=1:bW
        v = x(h,w,:);
        v = v(:);
        feat(:,off+w) = v;
    end
end
feat = single(feat);



function docs = get_images_docs(params, docs)

disp('* Getting images and resizing *');
for i=1:length(docs)
    docs(i).image = imread(docs(i).pathImage);
    H = docs(i).H;
    W = docs(i).W;
    res = mod(H,params.sbin);
    while res~=0
        H = H-res;
        res = mod(H,params.sbin);
    end
    res = mod(W,params.sbin);
    while res~=0
        W = W-res;
        res = mod(W,params.sbin);
    end
    difH = docs(i).H-H;
    difW = docs(i).W-W;
    padYini = floor(difH/2);
    padYend = difH-padYini;
    padXini = floor(difW/2);
    padXend = difW-padXini;
    im = docs(i).image(1+padYini:end-padYend,1+padXini:end-padXend);
    docs(i).image = im;
    docs(i).yIni = padYini+1;
    docs(i).xIni = padXini+1;
    [docs(i).H,docs(i).W] = size(docs(i).image);
end
end
function [docs, relevantBoxesByClass, numRelevantWordsByClass] = get_docs(params, classes)

disp('* Loading documents *');

file = params.fileDocuments;
if ~exist(file, 'file')
    disp('* Initializing test documents *');
    path = params.pathDocuments;
    d = dir([path '*.gtp']);
    
    numDocs = length(d);
    numRelevantWordsByClass = int32(zeros(length(classes),1));
    relevantBoxesByClass = cell(length(classes),numDocs);
    
    idxword = 1;
    for i=1:numDocs
        fid = fopen([path d(i).name], 'r');
        input = textscan(fid, '%d %d %d %d %s');
        nWords = length(input{1});
        for j=1:nWords
            words(j).loc = [input{1}(j) input{3}(j) input{2}(j) input{4}(j) i];
            words(j).gttext = input{5}{j};
            words(j).H = words(j).loc(4) - words(j).loc(3) + 1;
            words(j).W = words(j).loc(2) - words(j).loc(1) + 1;
            if isKey(classes,words(j).gttext)
                class = classes(words(j).gttext);
                words(j).class = class;
                numRelevantWordsByClass(class) = numRelevantWordsByClass(class)+1;
                relevantBoxesByClass{class,i} = [relevantBoxesByClass{class,i}; words(j).loc(1:4)]; 
            else
                words(j).class = -1;
            end
            words(j).globalIdx = idxword;
            idxword = idxword + 1;
        end
        docs(i).nWords = nWords;
        docs(i).words = words;
        pathImage = [params.pathImages d(i).name(1:end-3) 'tif'];
        im = imread(pathImage);
        [docs(i).H, docs(i).W] = size(im);
        docs(i).pathImage = pathImage;
        clear words;
    end
    save(file, 'docs', 'relevantBoxesByClass', 'numRelevantWordsByClass');
else
    load(file);
end
end
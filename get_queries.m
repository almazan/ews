function [queries, classes] = get_queries(params)

disp('* Loading queries *');

fileName = params.fileQueries;
if ~exist(fileName, 'file')
    disp('* Computing queries *');
    fileQueries=[params.pathQueries 'queries.gtp'];
    fid = fopen(fileQueries, 'r');
    
    input = textscan(fid, '%s %d %d %d %d %s');
    nWords = length(input{1});
    
    for j=1:nWords
        queries(j).pathIm = [params.pathImages input{1}{j}];
        queries(j).loc = [input{2}(j) input{4}(j) input{3}(j) input{5}(j)];
        queries(j).gttext = input{6}{j};
        queries(j).H = queries(j).loc(4) - queries(j).loc(3) + 1;
        queries(j).W = queries(j).loc(2) - queries(j).loc(1) + 1;
    end
    
    newClass = 1;
    queries(1).class = [];
    classes = containers.Map();
    for i=1:length(queries)
        % Determine the class of the query given the GT text
        if isKey(classes, queries(i).gttext)
            class = classes(queries(i).gttext);
        else
            class = newClass;
            newClass = newClass+1;
            classes(queries(i).gttext) = class;
        end
        queries(i).class = class;
    end
    %     classes.indexToName = classesNames;
    save(fileName, 'queries', 'classes');
else
    load(fileName);
end

end
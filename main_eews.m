% Function main_eews
% Jon Almazán and Albert Gordo
% 06/07/2012
% Main program to execute the complete Exemplar Word Spotting framework



%% Add libraries and define initial parameters
addpath('./util');
addpath('./util/jsgd-55/matlab');
addpath('./util/features_esvm');
addpath('./util/yael_matlab_linux64_v277');

params = get_initparams();
% Alternatively, you can use the following function to validate the parameters:
% params = validation_script();

%% Get queries
[queries, classes] = get_queries(params);

%% Get test documents
[docs, relevantBoxesByClass, numRelevantWordsByClass] = get_docs(params, classes);
params.numDocs = length(docs);
params.numNWords = ceil((params.numNWords/params.numDocs))*params.numDocs;


%% Compute features of the test documents
[docs, PCA, PQ_centroids] = compute_features_docs(params, docs);

mAP = zeros(length(queries),1);
scores = cell(length(queries),1);
resultLabels = cell(length(queries),1);
locWords = cell(length(queries),1);

%% Learn model and evaluate each query
% Load models
if exist(params.fileModels, 'file')
    load(params.fileModels);
else
    models = cell(length(queries),1);
end
for i = 80:length(queries)
    q = queries(i);
    class = q.class;
    nrelW = numRelevantWordsByClass(class);
    
    %% Learn model
    if isempty(models{i})
        models{i} = learn_model(params, q, docs, PCA, PQ_centroids);
    end
    
    %% Retrieve regions of the query
    [scores{i}, resultLabels{i}, locWords{i}] = eval_model(params, models{i}, docs, relevantBoxesByClass(q.class,:), PQ_centroids);
    
    if params.showResults
       save_result_images(q, i, locWords{i}, resultLabels{i}, docs, params);
    end
    
    %% Compute mAP
    mAP(i) = compute_mAP(resultLabels{i}, nrelW);
    fprintf('%20s (%5d): %2.2f (mAP) nRel: %d\n', q.gttext, i, mAP(i), nrelW);
end
if ~exist(params.fileModels, 'file')
    save(params.fileModels, 'models');
end

save(params.fileResults, 'mAP', 'scores', 'resultLabels', 'locWords');

fprintf('\n*** Final mAP: %f ***\n',mean(mAP));
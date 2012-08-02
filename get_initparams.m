function params = get_initparams()

params.dataset = 'GW';
params.showResults = 1;
params.numResultImages = 50;

% HOG parameters
params.descrType = 'hog_esvm';
params.sbin = 12;
params.obin = 9;
params.levels_per_scale = 1;
params.min_scale = 0.9;
params.max_scale = 1.1;
params.orig_scale_index = params.levels_per_scale + 1;
params.scales = unique([params.min_scale:(1-params.min_scale)/params.levels_per_scale:1 ...
    1:(params.max_scale-1)/params.levels_per_scale:params.max_scale]);

% Training parameters
params.rangeX = -10:2:10;
params.rangeY = -10:2:10;
% params.rangeX = -1:1:1;
% params.rangeY = -1:1:1;
params.numTrWords = length(params.rangeX)*length(params.rangeY);
params.propNWords = 64;
params.numNWords = params.numTrWords*params.propNWords;
params.svmlib = 'jsgd';

%% LB
% params.lambda = 1e-1;
% params.eta = 1e-3;
% params.epochs = 10;
% params.bias = 1;

%% GW
params.lambda = 1e-5;
params.eta = 1e-3;
params.epochs = 10;
params.bias = 1;

% Test parameters
params.thrWindows = 2000;
params.step = 1;
params.overlap = 0.5;
params.overlapnms = 0.2;

% PCA parameters
params.NPCA = 10000;
params.PCADIM = 16;
if params.PCADIM > 0
    params.dim = params.PCADIM;
else
    params.dim = 31;
end

% PQ parameters
params.DoPQ = 1;
params.tagpq = '';
if params.DoPQ
	params.tagpq = '_pq';
end
	
% Paths
params.pathImages = sprintf('datasets/%s/images/', params.dataset);
params.pathDocuments = sprintf('datasets/%s/documents/', params.dataset);
params.pathQueries = sprintf('datasets/%s/queries/', params.dataset);
params.pathData = 'data/';
if ~exist(params.pathData, 'dir')
    mkdir(params.pathData);
end
params.pathResultsParent = 'data/results';
if ~exist(params.pathResultsParent, 'dir')
    mkdir(params.pathResultsParent);
end
params.pathResults = sprintf('data/results/%s/',params.dataset);
if ~exist(params.pathResults, 'dir')
    mkdir(params.pathResults);
end
params.pathResultsImages = sprintf('%simages/',params.pathResults);
if ~exist(params.pathResultsImages, 'dir')
    mkdir(params.pathResultsImages);
end
params.fileFeatures = sprintf('data/%s_features_docs_sbin%d%s.mat',params.dataset,params.sbin,params.tagpq);
params.fileModels = sprintf('data/%s_queries_models_sbin%d%s.mat',params.dataset,params.sbin,params.tagpq);
params.fileResults = sprintf('%s%s_results_sbin%d%s.mat',params.pathResults,params.dataset,params.sbin,params.tagpq);
params.fileQueries = sprintf('data/%s_queries.mat',params.dataset);
params.fileDocuments = sprintf('data/%s_documents.mat',params.dataset);
end

function params = get_valparams()

params.dataset = 'GW';

% HOG parameters
params.descrType = 'hog_esvm';
params.sbin = 12;
params.obin = 9;


% Training parameters
params.rangeX = -10:2:10;
params.rangeY = -10:2:10;
params.numTrWords = length(params.rangeX)*length(params.rangeY);
params.propNWords = 64;
params.numNWords = params.numTrWords*params.propNWords;
params.svmlib = 'jsgd';
% params.lambda = 1e-5;
% params.eta = 1e-3;
% params.epochs = 10;
% params.bias = 1;

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
params.pathImages = sprintf('../Datasets/%s/images/', params.dataset);
params.pathDocuments = sprintf('../Datasets/%s/validation/documents/', params.dataset);
params.pathQueries = sprintf('../Datasets/%s/validation/queries/', params.dataset);
params.fileFeatures = sprintf('data/%s_features_docs_sbin%d%s_val.mat',params.dataset,params.sbin,params.tagpq);
params.fileModels = sprintf('data/%s_queries_models_sbin%d%s_val.mat',params.dataset,params.sbin,params.tagpq);
params.fileResults = sprintf('data/%s_results_sbin%d%s_val.mat',params.dataset,params.sbin,params.tagpq);
params.fileQueries = sprintf('data/%s_queries_val.mat',params.dataset);
params.fileDocuments = sprintf('data/%s_documents_val.mat',params.dataset);
end

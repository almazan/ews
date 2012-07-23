function params = validation_script()

% Validation parameters script for EEWS
% Jon Almazán and Albert Gordo
% 06/07/2012
% Script to validate the parameters of the Exemplar Word Spotting framework


%% Add libraries and define initial parameters
addpath('./util');
addpath('./util/jsgd-55/matlab');
addpath('./features_esvm');
addpath('./util/yael_matlab_linux64_v277');

params = get_valparams();

%% Get queries
[queries, classes] = get_queries(params);

%% Get test documents
[docs, relevantBoxesByClass, numRelevantWordsByClass] = get_docs(params, classes);
params.numDocs = length(docs);
params.numNWords = ceil((params.numNWords/params.numDocs))*params.numDocs;

%% Compute features of the test documents
[docs, PCA, PQ_centroids] = compute_features_docs(params, docs);

lambdaV = [1e-5 1e-3 1e-1];
etaV = [1e-5 1e-3 1e-1];
propNV = [64];
epochsV = [10];
biasV = [sqrt(0.5) sqrt(1) sqrt(2)];

mAPVal = zeros(length(lambdaV),length(etaV),length(propNV),length(epochsV),length(biasV));

for il=1:length(lambdaV)
    for ie=1:length(etaV)
        for in=1:length(propNV)
            for iep=1:length(epochsV)
                for ib=1:length(biasV)
                    
                    params.lambda = lambdaV(il);
                    params.eta = etaV(ie);
                    params.epochs = epochsV(iep);
                    params.propNWords = propNV(in);
                    params.numNWords = params.numTrWords*params.propNWords;
                    params.numNWords = ceil((params.numNWords/params.numDocs))*params.numDocs;
                    params.bias = biasV(ib);
                    
                    fprintf('l %1.7f - e %1.7f - n %4d - ep %5d - b %2.2f ...',lambdaV(il),etaV(ie),propNV(in),epochsV(iep),biasV(ib));
                    
                    mAP = zeros(length(queries),1);
                    %% Learn model and evaluate each query
                    parfor i = 1:length(queries)
                        q = queries(i);
                        class = q.class;
                        nrelW = numRelevantWordsByClass(class);
                        
                        %% Learn model
                        model = learn_model(params, q, docs, PCA, PQ_centroids);
                        
                        %% Retrieve regions of the query
                        [scs, resultLabels, lW] = eval_model(params, model, docs, relevantBoxesByClass(q.class,:), PQ_centroids);
                        
                        %% Compute mAP
                        mAP(i) = compute_mAP(resultLabels, nrelW);
                        
                    end
                    
                    mAPVal(il,ie,in,iep,ib) = mean(mAP);
                    fprintf(' %2.2f (mAP)\n', mean(mAP));
                end
            end
        end
    end
end

[v,idx] = max(mAPVal(:));
[il,ie,in,iep,ib] = ind2sub(size(mAPVal),idx);
params.lambda = lambdaV(il);
params.eta = etaV(ie);
params.epochs = epochsV(iep);
params.propNWords = propNV(in);
params.bias = biasV(ib);

fprintf('\nBest validation parameters - l %1.7f - e %1.7f - n %4d - ep %5d - b %2.2f : %2.2f (mAP)\n',lambdaV(il),etaV(ie),propNV(in),epochsV(iep),biasV(ib),mean(mAP));

save(params.fileResults, 'mAPVal');
end
function [mAP, rec] = compute_mAP(resultLabels, numRelevant)

% Compute the mAP
numRelevant = double(numRelevant);
precAt = cumsum(resultLabels)./[1:length(resultLabels)]';
mAP = sum(precAt.*resultLabels)/numRelevant;
rec= sum(resultLabels)/numRelevant;
end
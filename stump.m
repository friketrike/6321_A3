function [ Threshold, Dim, polarity ] = stump( Obs, C, W )
%STUMP Returns the best decision stump as a threshold-dimension pair for
%points contained in the Obs matrix given class label C and weights W 
%vectors
%   Obs should contain row entries for observations, where columns are
%   features. C is a vector containing binary classifications and W are the
%   weights associated with each Observation (eg weights assigned by a
%   boosting algorithm. The threshold is the point at which a decision
%   boundary perpendicular to the selected dimension should pass.
[m,d] = size(Obs);
% Normalize W, in case it hasn't been done
W = W/sum(W);
ObsIDX_W_C_E = zeros(m,d,5);
%Order each feature, superimpose a matrix of row indices and then
%superimpose row weights, row classes and errors over it .
[ObsIDX_W_C_E(:,:,1), ObsIDX_W_C_E(:,:,2)] = sort(Obs);
ObsIDX_W_C_E(:,:,3) = W(ObsIDX_W_C_E(:,:,2));
ObsIDX_W_C_E(:,:,4) = (C(ObsIDX_W_C_E(:,:,2))*2)-1;
ObsIDX_W_C_E(:,:,5) = NaN*(ones(m,d));
diff_neighbors = [ones(1,d);0.5*abs(diff(ObsIDX_W_C_E(:,:,4)))];

for n = 1:d
    for i = 1:m
        if diff_neighbors(i,n) == 1
            yhat = [-1*ones(length(1:i-1),1);ones(length(i:m),1)];
            ObsIDX_W_C_E(i,n,5) = sum(abs( ...
                (yhat ~= ObsIDX_W_C_E(:,n,4)) ...
                .*ObsIDX_W_C_E(:,n,3)));
        end
    end
end

% ObsIDX_W_C_E % just a print for debugging purposes

% get the error farthest from 0.5, the most informative split
[mx,idx] = max(abs(0.5-ObsIDX_W_C_E(:,:,5)));

% we need to know along which direction it happened
[~, Dim] = max(mx);
 
% recover the true index
idx = idx(Dim);

% polarity 1 means that positive class observations are contained at 
% higher values for that feature, polarity -1 means positive observations
% live at lower values than the threshold
if ObsIDX_W_C_E(idx,Dim,5) > 0.5
    polarity = -1;
else
    polarity = 1;
end

% Now calculate where the split happens, the midpoint between two adjacent 
% points on that feature dimension, repeat the first point so that decision
% at the lower bound is just the smallest value of the feature space
feature_vals = [ObsIDX_W_C_E(1,Dim, 1);ObsIDX_W_C_E(:,Dim, 1)];
Threshold = mean(feature_vals(idx:idx+1));
end


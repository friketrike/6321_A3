function [ Threshold, Dim, polarity, err ] = stump( X, y, W )
%STUMP Returns the best decision stump as a threshold-dimension pair for
%points contained in the Obs matrix given class label C and weights W 
%vectors
%   Obs should contain row entries for observations, where columns are
%   features. C is a vector containing binary classifications and W are the
%   weights associated with each Observation (eg weights assigned by a
%   boosting algorithm. The threshold is the point at which a decision
%   boundary perpendicular to the selected dimension should pass.
    [m,d] = size(X);
    % Normalize W, in case it hasn't been done
    W = W/sum(W);
    Obs = zeros(m,d);
    IDXs = zeros(m,d);
    % Order each feature, keep a matrix of row indices per feature and then
    % keep row classes and errors.
    [Obs, IDXs] = sort(X);
    Ys = (y(IDXs)*2)-1;
    E = NaN*(ones(m,d));
    % we can only have a split between neighbours of different classes, don't
    % bother checking between neighbours of the same class
    diff_neighbors = [2*ones(1,d);0.5*abs(diff(Ys))];

    for n = 1:d
        for i = 1:m
            %if diff_neighbors(i,n) == 1
                C_hat = X(:,n) >= Obs(i,n);
                err = sum((C_hat ~= y).*W);
                E(i,n) = err;
            %end
        end
    end

    % get the error farthest from 0.5, the most informative split
    [mx,idx] = max(abs(0.5-E));

    % we need to know along which direction it happened
    [~, Dim] = max(mx);

    % recover the true index
    idx = idx(Dim);

    err = E(idx,Dim);

    % polarity 1 means that positive class observations are contained at 
    % higher values for that feature, polarity -1 means positive observations
    % live at lower values than the threshold
    if err > 0.5
        polarity = -1;
        err = 1 - err;
    else
        polarity = 1;
    end

    % Now calculate where the split happens, the midpoint between two adjacent 
    % points on that feature dimension, repeat the first point so that decision
    % at the lower bound is just the smallest value of the feature space
    cut_point = Obs(idx,Dim);
    feature_vals = unique(Obs(:,Dim));
    the_real_idx = cut_point == feature_vals;
    feature_vals = [(feature_vals(1) - feature_vals(2)); feature_vals];
    midpoints = feature_vals + 0.5*([diff(feature_vals);0]);
    Threshold = midpoints(the_real_idx);
end


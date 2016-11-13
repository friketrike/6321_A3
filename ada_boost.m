function [h, alpha, W] = ada_boost(X, y, W)
%ADA_BOOST takes a series of observations (each observation as a row), labels
%and a vector W of weights for each observation. It returns a series
%[thr,dim,polarity] tuples per stump, their associated weights and a a two-column 
%error matrix where each row is associatedassociated with each iteration and 
%training and test errors populate the first and second columns respectively.
% TODO Detailed info?


    [ Threshold, Dim, polarity, err ] = stump(X, y, W);
    h = [Threshold, Dim, polarity];
    
    y_hat = X(:,h(2)) > h(1);
    if h(3) == -1
        y_hat = ~y_hat;
    end
    
    alpha = 0.5 * log((1-err)/err);
    % a vector containing 1 for correctly classified entries given h(round)
    % and -1 for misclassified entries.
    classifiedRightOrNot = ((2*y)-1).*((2*y_hat)-1);
    W = W .* exp(-alpha*(classifiedRightOrNot));
    W = W./sum(W);
end    


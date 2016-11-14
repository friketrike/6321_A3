% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 3, due November 17

function [h, alpha, W] = ada_boost(X, y, W)
%ADA_BOOST takes an observation matrix X (each observation as a row), labels
%y and a vector W of weights for each observation. It returns a series
%h [thr,dim,polarity] tuples per stump, their associated weights and a a 
%two-column error matrix.
%   for the error matrix, row is associated with each iteration and training 
%   and test errors populate the first and second columns respectively.

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


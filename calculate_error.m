% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 3, due November 17

function [ err, y_hat ] = calculate_error( X, y, h, alphas )
%CALCULATE_ERROR calculates the error given a dataset X, its target labels
%y and the ensemble adaboost predictor h, alpha.
%   X are the observations given in row form, y the labels given as a
%   column vector, h tha matrix of [threshold, dim, polarity] weak
%   classifiers and alphas is the column vector of weights associated with
%   each weak classifier tuple.

y_hat = zeros(length(y), 1);

m = length(y);
for i = 1:m
    for k = 1:length(alphas)
        threshold = h(k, 1);
        dim = h(k, 2);
        pol = h(k, 3);
        contrib_k = (2*(pol*X(i, dim) >= pol*threshold)-1)*alphas(k);
        y_hat(i) = y_hat(i) + contrib_k;
    end
    y_hat(i) = y_hat(i) >= 0;
end

err = sum((y_hat > 0) ~= y)/m;
end


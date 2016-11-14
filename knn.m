% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 3, due November 17

function [yi_hats, IDXs] = knn(X, y, xi, max_k)
%KNN returns the class estimates for xi, given xi's 1:max_k nearest neighbours.
%It also returns the indexes of said neighbours while requiring a 
%row-entry matrix of observations X and a corresponding column of labels y.
    m = length(y);
    dist = zeros(m);
    for i = 1:m
        dist(i) = norm(xi - X(i,:));
    end
    [~, idx] = sort(dist);
    for k = 1:max_k
        IDXs = idx(1:k);
        yi_hats(k) = round(sum(y(IDXs))/k);
    end
end

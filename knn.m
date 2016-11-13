function [yi_hat, IDXs] = knn(X, y, xi, k)
%KNN returns the class estimate for xi, given xi's k nearest neighbours.
%It also returns the indexes of said neighbours while requiring a 
%row-entry matrix of observations X and a corresponding column of labels y.
    m = length(y);
    dist = zeros(m);
    for i = 1:m
        dist(i) = norm(xi - X(i,:));
    end
    [~, idx] = sort(dist);
    yi_hat = round(sum(y(1:k))/k);
    IDXs = idx(1:k);
end

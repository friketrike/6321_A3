% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 3, due November 17

X = load('wpbcx.dat');
y = load('wpbcy.dat');

num_folds = 10;

folds_info = cvpartition(length(y), 'kfold', num_folds);
folds_idx = randperm(length(y));

iters = 50;
max_k = 50;

errs_ada = zeros(iters, num_folds, 2);
errs_knn = zeros(max_k, num_folds, 2);

for fold = 1:num_folds
    disp(sprintf('Performing %d-fold CV, fold: %d', num_folds, fold));
    idxs_prev = 1:sum(folds_info.TestSize(1:(fold-1)));
    if ~isempty(idxs_prev)
        offset = idxs_prev(end);
    else
        offset = 0;
    end
    idxs_xcl = (1:folds_info.TestSize(fold))+offset;
    idx_after_skip = length(y)-(sum(folds_info.TestSize((fold+1):end))-1);
    idxs_next = idx_after_skip:length(y);
    X_train = X(folds_idx([idxs_prev, idxs_next]), :);
    X_test = X(folds_idx(idxs_xcl), :);
    y_train = y(folds_idx([idxs_prev, idxs_next]));
    y_test = y(folds_idx(idxs_xcl));
    
    h = zeros(iters, 3);
    alphas = zeros(iters,1);

    m = length(y_train);
    W = ones(m, 1)/m;
    %%% Adaboost loop %%%
    for i = 1:iters
        % keep the user informed on longer runs
        if mod(i, 100) == 0
            disp(sprintf('\tworking on iter %d', i));
        end
        [h(i,:), alphas(i), W] = ada_boost(X_train, y_train, W);
        errs_ada(i, fold, 1) = calculate_error(X_train, y_train, ...
                                           h(1:i, :), alphas(1:i));
        errs_ada(i, fold, 2) = calculate_error(X_test, y_test, ...
                                           h(1:i, :), alphas(1:i));
    end
    %%% KNN loop %%%
    for l = 1:m
        % remove point i from the training set
        X_train_temp = X_train;
        y_train_temp = y_train;
        X_train_temp(l,:) = [];
        y_train_temp(l) = [];
        yi_hats = knn(X_train_temp, y_train_temp, X_train(l, :),max_k);
        for k = 1:max_k
            if yi_hats(k) ~= y_train(l)
                errs_knn(k, fold, 1) = errs_knn(k, fold, 1) + 1/m;
            end
        end
    end
    n = length(y_test);
    for l = 1:n
        yi_hats = knn(X_train, y_train, X_test(l, :),max_k);
        for k = 1:max_k
            if yi_hats(k) ~= y_train(l)
                errs_knn(k, fold, 2) = errs_knn(k, fold, 2) + 1/n;
            end
        end
    end
end

figure(1)
plot(mean(errs_ada(:,:,1), 2))
hold on;
plot(mean(errs_ada(:,:,2), 2))
legend('training error', 'testing error', 'location', 'east')
title('10-fold adaboost')
xlabel('iterations')
ylabel('error')
hold off
saveas(gcf, 'ada-plot.pdf');

figure(2)
plot(mean(errs_knn(:,:,1), 2))
hold on;
plot(mean(errs_knn(:,:,2), 2))
legend('training error', 'testing error')
title('10-fold knn')
xlabel('k')
ylabel('error')
hold off
saveas(gcf, 'knn-plot.pdf')

% how well did we do?
disp('Here''s how well we did:')
disp(sprintf('best prediction given on test data by adaboost:\n\t%d',...
              min(mean(errs_ada(:,:,2), 2)) ));
disp(sprintf('best prediction given on test data by knn:\n\t%d',...
              min(mean(errs_knn(:,:,2), 2)) ));              
disp(sprintf('empirical ratio of class 1 to class 0:\n\t%d',...
              sum(y)/length(y) ));


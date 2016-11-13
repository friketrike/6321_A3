
% Testing with different cases...
% % X = rand(50,2);
% % y = (rand(50,1) + X(:,1) - 0.5 * X(:,2)) < 0.5;

% % load('pima-indians-diabetes.data')
% % X = pima_indians_diabetes(:, 1:end-1);
% % y = pima_indians_diabetes(:,end);

X = load('wpbcx.dat');
y = load('wpbcy.dat');

num_folds = 10;

folds_info = cvpartition(length(y), 'kfold', num_folds);
folds_idx = randperm(length(y));

iters = 200;

errs = zeros(iters, num_folds, 2);

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
    for i = 1:iters
        % keep the user informed on longer runs
        if mod(i, 100) == 0
            disp(sprintf('\tworking on iter %d', i));
        end
        [h(i,:), alphas(i), W] = ada_boost(X_train, y_train, W);
        errs(i, fold, 1) = calculate_error(X_train, y_train, ...
                                           h(1:i, :), alphas(1:i));
        errs(i, fold, 2) = calculate_error(X_test, y_test, ...
                                           h(1:i, :), alphas(1:i));
    end
end

figure(1)
plot(mean(errs(:,:,1), 2))
hold on;
plot(mean(errs(:,:,2), 2))
legend('training error', 'testing error')
title('10-fold adaboost')
xlabel('iterations')
ylabel('error')
hold off

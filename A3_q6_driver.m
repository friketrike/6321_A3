

% dumb sample usage for the time being...
Obs = rand(20,2);
C = (rand(20,1) + Obs(:,1) - 0.5 * Obs(:,2)) < 0.5;
%Obs = [0,0;1,1;1,0;0,1];
%C = logical([0;1;1;1]);
% weights = ones(50,1);
% weights = weights/sum(weights);
% [t,d, polarity, e] = stump(someData, labels, weights);
% 
% plot(someData(labels,1), someData(labels,2), '.g')
% hold on
% plot(someData(~labels,1), someData(~labels,2), '.r')
% 
% if d == 1
%     plot([t,t],[min(someData(:,1)),max(someData(:,1))]);
% else
%     plot([min(someData(:,1)),max(someData(:,1))], [t,t]);
% end
% 
% hold off
%load('pima-indians-diabetes.data')
%Obs = pima_indians_diabetes(:, 1:2);
%C = pima_indians_diabetes(:,end);
%%Just putting this in from the last assignment, TODO taylor it to our needs here...
%TODO bring plotting outside of ada_boost
%TODO figure out error not converging
% get data...
X = load('wpbcx.dat');
X = [X, ones(size(X,1),1)];
y = load('wpbcy.dat');

num_folds = 10;

folds_info = cvpartition(length(y), 'kfold', num_folds);
folds_idx = randperm(length(y));

[h, alphas, errs] = ada_boost(Obs, C, 100);

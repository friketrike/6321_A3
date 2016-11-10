

% dumb sample usage for the time being...
Obs = rand(10,2);
C = (rand(10,1) + Obs(:,1) - 0.5 * Obs(:,2)) < 0.5;
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

[h, alphas, errs] = ada_boost(Obs, C, 10);

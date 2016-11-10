
function [h, alphas, errs] = ada_boost(Obs, C, rounds)
%ADA_BOOST takes a series of observations (each observation as a row), labels
%and an optional argument giving an upper bound to iterations. It returns a series
%{thr,dim,polarity} tuples per stump, their associated weights and a a two-column 
%error matrix where each row is associatedassociated with each iteration and 
%training and test errors populate the first and second columns respectively.

% default early termination
if nargin < 3 || isempty(rounds)
    rounds = 2; % TODO, add more
end

% container for Threshold, Dim, polarity tuples
h = zeros(rounds, 3); 
alphas = zeros(rounds, 1);
errs = zeros(rounds, 2);

m = length(C);
W = ones(m,1)/m;

% TODO partition, loop over partitions 
figure(1)
plot(Obs(logical(C),1), Obs(logical(C),2), 'og')
hold on
plot(Obs(~logical(C),1), Obs(~logical(C),2), 'xr')

for round = 1:rounds % this or a function of divergence of test and train errs
    % TODO change Obs for Obs_training
    [ Threshold, Dim, polarity, err ] = stump(Obs, C, W);
    h(round, :) = [Threshold, Dim, polarity];
    
    C_hat = Obs(:,h(round, 2)) > h(round, 1);
    if h(round, 3) == -1
        C_hat = ~C_hat;
    end
    
    alphas(round) = 0.5 * log((1-err)/err);
    % a vector containing 1 for correctly classified entries given h(round)
    % and -1 for misclassified entries.
    classifiedRightOrNot = ((2*C)-1).*((2*C_hat)-1);
    W = W .* exp(-alphas(round)*(classifiedRightOrNot));
    W = W./sum(W);
    figure(2)
    plot(W)
    % get new h
    % calculate train error given series of alphais his
    % calculate test error given series of alphais his
    
    if polarity == 1
        colour = 'k';
    else
        colour = ' b'; 
    end
    figure(1)
    hold on;
    if Dim == 1
        plot([Threshold,Threshold],[min(Obs(:,2)),max(Obs(:,2))], colour);
    else
        plot([min(Obs(:,1)),max(Obs(:,1))], [Threshold,Threshold], colour);
    end
    if err == 0
        alphas(round:end) = [];
        errs(round:end, :) = [];
        h(round:end, :) = [];
        continue;
    else
        errs(round, 1) = err;
    end
    disp({round, err, alphas(round)})
    pause(0.05)
%
end    
hold off
end

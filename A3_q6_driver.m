

% dumb sample usage for the time being...
someData = rand(50,2);
labels = rand(50,1)>0.5;
weights = ones(50,1);
weights = weights/sum(weights);
[t,d, polarity] = stump(someData, labels, weights);

plot(someData(labels,1), someData(labels,2), '.g')
hold on
plot(someData(~labels,1), someData(~labels,2), '.r')

if d == 1
    plot([t,t],[min(someData(:,1)),max(someData(:,1))]);
else
    plot([min(someData(:,1)),max(someData(:,1))], [t,t]);
end

hold off
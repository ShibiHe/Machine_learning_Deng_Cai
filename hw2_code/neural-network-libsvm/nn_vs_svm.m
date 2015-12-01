load('digit_data', 'X', 'y');
load('weights', 'Theta1', 'Theta2');

p = feedforward(Theta1, Theta2, X);
fprintf('Error rate for NN is %f.\n', length(find(p ~= y))/length(p));

%%
train_X = X(:, 1:2500);
train_y = y(1:2500);
test_X = X(:, 2501:end);
test_y = y(2501:end);

% YOUR CODE HERE
% Trainning and testing using one-vs-all with LIBLINEAR
label = zeros(size(Theta2, 2), size(train_y, 2));
l = zeros(size(Theta2, 2), size(test_y, 2));
decision_value = zeros(size(Theta2, 2), size(test_y, 2));

for i = 1:10
    label(i, :) = train_y == i;
    model(i) = train(label(i, :)', sparse(train_X'), '-q -s 2 -B 1');
    [l(i, :), ~, decision_value(i, :)] = predict(test_y', sparse(test_X'), model(i), '-q');
end

decision_value(8, :) = -decision_value(8, :); % 8 is the first
%
[~, index] = min(decision_value);
E_svm = sum(index ~= test_y) / size(test_y, 2);
fprintf('Error rate for SVM is %f.\n', E_svm);
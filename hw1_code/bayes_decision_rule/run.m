% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);
[C, N] = size(train_x);

%% Part1 likelihood: 
l = likelihood(train_x);

figure(1);
bar(range(1):range(2), l');
xlabel('x');
ylabel('P(x|\omega)');
axis([range(1) - 1, range(2) + 1, 0, 0.5]);
legend('\omega1','\omega2');

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule
e = abs(floor(l - repmat(max(l,[],1), [C, 1])));
err = sum(sum(test_x .* e))/sum(sum(test_x));
fprintf(1, 'test error using maximum likelihood decision rule: %6.2f\n', err);


%% Part2 posterior:
p = posterior(train_x);

figure(2);
bar(range(1):range(2), p');
xlabel('x');
ylabel('P(\omega|x)');
axis([range(1) - 1, range(2) + 1, 0, 1.2]);
legend('\omega1','\omega2');

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
e = abs(floor(p - repmat(max(p,[],1), [C, 1])));
err = sum(sum(test_x .* e))/sum(sum(test_x));
fprintf(1, 'test error using optimal Bayesian decision rule: %6.2f\n', err);

%% Part3 risk:
risk = [0, 1; 2, 0];
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
r = [sum(repmat(risk(1,:), [N, 1])' .* p, 1); sum(repmat(risk(2,:), [N, 1])' .* p, 1)];
e = abs(floor(r - repmat(max(r,[],1), [C, 1])));
err = sum(sum(test_x .* r .* e));
fprintf(1, 'minimum risk: %6.2f\n', err);

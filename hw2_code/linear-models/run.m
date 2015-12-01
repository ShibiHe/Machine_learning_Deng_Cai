% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest=10000;

totalIter=0;
E_test_tot=0;
E_train_tot=0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain+nTest);
    X_test=X(:,nTrain+1:end);
    y_test=y(:,nTrain+1:end);
    X=X(:,1:nTrain);
    y=y(:,1:nTrain);
    
    [w_g, iter] = perceptron(X, y);
    % X=X(:,1:10);
    % y=y(1:10);
    pred=sign(w_g'*[ones(1,size(X,2));X]);
    pred=pred-y;
    error=size(pred(pred~=0),2);
    E_train=error/size(X,2);

    pred=sign(w_g'*[ones(1,size(X_test,2));X_test]);
    pred=pred-y_test;
    error=size(pred(pred~=0),2);
    E_test=error/size(X_test,2);
    fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
    E_test_tot=E_test_tot+E_test;
    E_train_tot=E_train_tot+E_train;
    totalIter=totalIter+iter;
    % Compute training, testing error
    % Sum up number of iterations
end

% % this is for gradient check.
% theta=randn(size(X,1)+1,1);
% disp(grad_check(@perceptron_test,theta,5,X,y));
fprintf('expected E_train is %f, expected E_test is %f.\n', E_train_tot/nRep, E_test_tot/nRep);
fprintf('Average number of iterations is %d\n', totalIter/nRep);
% plotdata(X, y, w_f, w_g, 'Pecertron');


%% Part2: Preceptron: Non-linearly separable case
nTrain = 1000; % number of training data
[X, y, w_f] = mkdata(nTrain, 'noisy');
[w_g, iter] = perceptron(X, y);
fprintf('Average number of iterations is %d.\n', iter);
plotdata(X, y, w_f, w_g, 'Pecertron');


%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest=10000;
E_test_tot=0;
E_train_tot=0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain+nTest);
    X_test=X(:,nTrain+1:end);
    y_test=y(:,nTrain+1:end);
    X=X(:,1:nTrain);
    y=y(:,1:nTrain);
    
    w_g = linear_regression(X, y);
    % Compute training, testing error
    pred=sign(w_g'*[ones(1,size(X,2));X]);
    pred=pred-y;
    error=size(pred(pred~=0),2);
    E_train=error/size(X,2);

    pred=sign(w_g'*[ones(1,size(X_test,2));X_test]);
    pred=pred-y_test;
    error=size(pred(pred~=0),2);
    E_test=error/size(X_test,2);
    fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
    E_train_tot=E_train_tot+E_train;
    E_test_tot=E_test_tot+E_test;
end
fprintf('expected E_train is %f, expected E_test is %f.\n', E_train_tot/nRep, E_test_tot/nRep);


plotdata(X, y, w_f, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest=10000;
E_test_tot=0;
E_train_tot=0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain+nTest,'noisy');
    X_test=X(:,nTrain+1:end);
    y_test=y(:,nTrain+1:end);
    X=X(:,1:nTrain);
    y=y(:,1:nTrain);
    
    w_g = linear_regression(X, y);
    % Compute training, testing error
    pred=sign(w_g'*[ones(1,size(X,2));X]);
    pred=pred-y;
    error=size(pred(pred~=0),2);
    E_train=error/size(X,2);

    pred=sign(w_g'*[ones(1,size(X_test,2));X_test]);
    pred=pred-y_test;
    error=size(pred(pred~=0),2);
    E_test=error/size(X_test,2);
    fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
    E_test_tot=E_test_tot+E_test;
    E_train_tot=E_train_tot+E_train;
end
fprintf('expected E_train is %f, expected E_test is %f.\n', E_train_tot/nRep, E_test_tot/nRep);

plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');
w_g = linear_regression(X, y);
pred=sign(w_g'*[ones(1,size(X,2));X]);
pred=pred-y;
error=size(pred(pred~=0),2);
E_train=error/size(X,2);

pred=sign(w_g'*[ones(1,size(X_test,2));X_test]);
pred=pred-y_test;
error=size(pred(pred~=0),2);
E_test=error/size(X_test,2);
% Compute training, testing error
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
% CHANGE THIS LINE TO DO TRANSFORMATION
X(3,:) = X(1,:).*X(2,:);
X(4,:) = X(1,:).*X(1,:);
X(5,:) = X(2,:).*X(2,:);
X_test = X_test; % CHANGE THIS LINE TO DO TRANSFORMATION
X_test(3,:) = X_test(1,:).*X_test(2,:);
X_test(4,:) = X_test(1,:).*X_test(1,:);
X_test(5,:) = X_test(2,:).*X_test(2,:);

w_g = linear_regression(X, y);

pred=sign(w_g'*[ones(1,size(X,2));X]);
pred=pred-y;
error=size(pred(pred~=0),2);
E_train=error/size(X,2);

pred=sign(w_g'*[ones(1,size(X_test,2));X_test]);
pred=pred-y_test;
error=size(pred(pred~=0),2);
E_test=error/size(X_test,2);

% Compute training, testing error
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest=10000;
% this is for gradient check.
% [X, y, w_f] = mkdata(nTrain);
% theta=randn(size(X,1),1);
% disp(grad_check(@logistic_regression_test,theta,5,X,y));
E_test_tot=0;
E_train_tot=0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain+nTest);
    y(y==-1)=0;
    X_test=X(:,nTrain+1:end);
    y_test=y(:,nTrain+1:end);
    X=X(:,1:nTrain);
    y=y(:,1:nTrain);
    
    w_g = logistic(X, y);
    % Compute training, testing error
    pred=h_theta(w_g,[ones(1,size(X,2));X]);
    pred(pred>0.5)=1;
    pred(pred<=0.5)=0;
    pred=pred-y;
    error=size(pred(pred~=0),2);
    E_train=error/size(X,2);

    pred=h_theta(w_g,[ones(1,size(X_test,2));X_test]);
    pred(pred>0.5)=1;
    pred(pred<=0.5)=0;
    pred=pred-y_test;
    error=size(pred(pred~=0),2);
    E_test=error/size(X_test,2);
    E_train_tot=E_train_tot+E_train;
    E_test_tot=E_test_tot+E_test;
    fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
end
fprintf('expected E_train is %f, expected E_test is %f.\n', E_train_tot/nRep, E_test_tot/nRep);

% plotdata(X, y, w_f, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10000; % number of training data
E_train_tot=0;
E_test_tot=0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain+nTest,'noisy');
    X_o=X;
    y_o=y;
    y(y==-1)=0;
    X_test=X(:,nTrain+1:end);
    y_test=y(:,nTrain+1:end);
    X=X(:,1:nTrain);
    y=y(:,1:nTrain);
    
    w_g = logistic(X, y);
    % Compute training, testing error
    pred=h_theta(w_g,[ones(1,size(X,2));X]);
    pred(pred>0.5)=1;
    pred(pred<=0.5)=0;
    pred=pred-y;
    error=size(pred(pred~=0),2);
    E_train=error/size(X,2);

    pred=h_theta(w_g,[ones(1,size(X_test,2));X_test]);
    pred(pred>0.5)=1;
    pred(pred<=0.5)=0;
    pred=pred-y_test;
    error=size(pred(pred~=0),2);
    E_test=error/size(X_test,2);
    fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
    E_train_tot=E_train_tot+E_train;
    E_test_tot=E_test_tot+E_test;
%     plotdata(X_o, y_o, w_f, w_g, 'Logistic Regression: noisy');
end
fprintf('expected E_train is %f, expected E_test is %f.\n', E_train_tot/nRep, E_test_tot/nRep);
%fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
% plotdata(X, y, w_f, w_g, 'Logistic Regression: noisy');

%% Part8: SVM
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest=10000;
totalSv=0;
E_train_tot=0;
E_test_tot=0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain+nTest);
    X_test=X(:,nTrain+1:end);
    y_test=y(:,nTrain+1:end);
    X=X(:,1:nTrain);
    y=y(:,1:nTrain);
    [w_g, num_sc] = svm(X, y);
    totalSv=totalSv+num_sc;
    % Compute training, testing error
    % Sum up number of support vectors
    
    pred=sign(w_g'*[ones(1,size(X,2));X]);
    pred=pred-y;
    error=size(pred(pred~=0),2);
    E_train=error/size(X,2);

    pred=sign(w_g'*[ones(1,size(X_test,2));X_test]);
    pred=pred-y_test;
    error=size(pred(pred~=0),2);
    E_test=error/size(X_test,2);
    
    E_train_tot=E_train_tot+E_train;
    E_test_tot=E_test_tot+E_test;
    fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
end
fprintf('expected E_train is %f, expected E_test is %f.\n', E_train_tot/nRep, E_test_tot/nRep);
fprintf('average number of support vectors is%d\n',totalSv/nRep);
plotdata(X_test, y_test, w_f, w_g, 'SVM');

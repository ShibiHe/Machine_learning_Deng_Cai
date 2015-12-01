%% Ridge Regression
load('digit_train', 'X', 'y');
% show_digit(X);
X=mapstd(X')';

% Do feature normalization


% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
validation_error=100000;
for i = 1:length(lambdas)
    fprintf('testing lambda:%d\n',i);
    E_val = 0;
    for j = 1:size(X, 2)
        % take point j out of X
        X_ = [X(:,1:j-1), X(:,j+1:end)]; y_ = [y(1,1:j-1), y(1,j+1:end)]; 
            w = ridge(X_, y_, lambdas(i));
            pred=sign(w'*[X(:,j);1]);
            pred=pred-y(j);
            if pred~=0
                E_val = E_val + 1;
            end
    end
    % Update lambda according validation error
    disp(E_val);
    if E_val<=validation_error
        validation_error=E_val;
        lambda=lambdas(i);
    end
end

% Compute training error
w = ridge(X, y, lambda);
pred=sign(w'*[X;ones(1,size(X,2))]);
pred=pred-y;
error=size(pred(pred~=0),2);
E_train=error/size(X,2);
fprintf('lambda=%f sigma w^2=%f\n',lambda,sum(w.^2));
fprintf('training error is%f\n',E_train);

lambda2=0;
w_0 = ridge(X, y, lambda2);
pred=sign(w_0'*[X;ones(1,size(X,2))]);
pred=pred-y;
error=size(pred(pred~=0),2);
E_train=error/size(X,2);
fprintf('lambda=%f sigma w^2=%f\n',lambda2,sum(w_0.^2));
fprintf('training error is%f\n',E_train);

load('digit_test', 'X_test', 'y_test');
% Do feature normalization
X_test=mapstd(X_test')';
% Compute test error
pred=sign(w'*[X_test;ones(1,size(X_test,2))]);
pred=pred-y_test;
error=size(pred(pred~=0),2);
E_test=error/size(X_test,2);
fprintf('lambda=%f testing error is%f\n',lambda,E_test);

pred=sign(w_0'*[X_test;ones(1,size(X_test,2))]);
pred=pred-y_test;
error=size(pred(pred~=0),2);
E_test=error/size(X_test,2);
fprintf('lambda=%f testing error is%f\n',lambda2,E_test);

%% Logistic
load('digit_train', 'X', 'y');
load('digit_test', 'X_test', 'y_test');
X=mapstd(X')';
X_test=mapstd(X_test')';
y(y==-1)=0;
y_test(y_test==-1)=0;


% I have done the LOOCV and find lambda=10 is the best
lambda=10;
% Compute training error
w = logistic_r(X, y, lambda);
pred=h_theta(w,[X;ones(1,size(X,2))]);
pred(pred>0.5)=1;
pred(pred<=0.5)=0;
pred=pred-y;
error=size(pred(pred~=0),2);
E_train=error/size(X,2);
fprintf('lambda=%f sigma w^2=%f\n',lambda,sum(w.^2));
fprintf('training error is%f\n',E_train);

lambda2=0;
w_0 = logistic_r(X, y, lambda2);
pred=h_theta(w_0,[X;ones(1,size(X,2))]);
pred(pred>0.5)=1;
pred(pred<=0.5)=0;
pred=pred-y;
error=size(pred(pred~=0),2);
E_train=error/size(X,2);
fprintf('lambda=%f sigma w^2=%f\n',lambda2,sum(w_0.^2));
fprintf('training error is%f\n',E_train);

% Compute test error
pred=h_theta(w,[X_test;ones(1,size(X_test,2))]);
pred(pred>0.5)=1;
pred(pred<=0.5)=0;
pred=pred-y_test;
error=size(pred(pred~=0),2);
E_test=error/size(X_test,2);
fprintf('lambda=%f testing error is%f\n',lambda,E_test);

pred=h_theta(w_0,[X_test;ones(1,size(X_test,2))]);
pred(pred>0.5)=1;
pred(pred<=0.5)=0;
pred=pred-y_test;
error=size(pred(pred~=0),2);
E_test=error/size(X_test,2);
fprintf('lambda=%f testing error is%f\n',lambda2,E_test);
disp('finished');

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
validation_error=100000;
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:20:size(X, 2)
        % take point j out of X
        X_ = [X(:,1:j-1), X(:,j+20:end)]; y_ = [y(1,1:j-1), y(1,j+20:end)]; 
            w = logistic_r(X_, y_, lambdas(i));
            pred=h_theta(w,[X(:,j:j+19);ones(1,20)]);
            pred(pred>0.5)=1;
            pred(pred<=0.5)=0;
            pred=pred-y(j:j+19);
            E_val=E_val+sum(pred~=0);
    end
    % Update lambda according validation error
    disp(E_val);
    if E_val<=validation_error
        validation_error=E_val;
        lambda=lambdas(i);
    end
end


%% SVM with slack variable

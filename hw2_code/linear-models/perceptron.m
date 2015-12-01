function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
X=[ones(1,size(X,2));X];
[P,N]=size(X);
w=randn(P,1)*0.01;

iter=0; eta=1;
oldD=1000000;
for i=1:1000000000
    iter=iter+1;
    pred=w'*X;
    D=0;
    Grad=zeros(P,1);
    mis=zeros(1,N);
    mis=logical(mis);
    for j=1:size(pred,2)
        if pred(j)*y(j)<0
            D=D-pred(j)*y(j);
            mis(j)=1;
        end
    end
    mis=~mis;
    if abs(D-oldD)<0.01
        break;
    else
        oldD=D;
    end
    X_grad=X;
    X_grad(:,mis)=0;
    Grad=X_grad*y';
    w=w+eta*Grad;
    if mod(i,1000)==0
        eta=eta*0.8;
        fprintf('iter=%d cost=%f\n',iter,D);
    end
end
end

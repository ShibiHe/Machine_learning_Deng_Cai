function [ f,g ] = perceptron_test(theta,X,y )
% this is used for gradient check.
%PERCEPTRON_TEST Summary of this function goes here
%   Detailed explanation goes here

X=[ones(1,size(X,2));X];
[P,N]=size(X);
pred=theta'*X;
f=0;
g=zeros(P,1);
mis=zeros(1,N);
mis=logical(mis);
for j=1:size(pred,2)
    if pred(j)*y(j)<0
        f=f-pred(j)*y(j);
        mis(j)=1;
    end
end
mis=~mis;
X_grad=X;
X_grad(:,mis)=0;
g=-X_grad*y';
end


function [f,g] = logistic_regression_test(theta, X,y)
y(y==-1)=0;
f=log(h_theta(theta,X))*y'+log(1-h_theta(theta,X))*(1-y)';
f=-f;
Xt=h_theta(theta,X)-y;
g=X*Xt';
end
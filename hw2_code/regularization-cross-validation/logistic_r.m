function w = logistic_r(X, y, lambda)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P,N]=size(X);
X=[X;ones(1,N)];
w=randn(P+1,1)*0.01;

theta=w;
eta=min([1/lambda*0.5, 0.1]);
iter=0;
oldf=100000;
for i=1:1000000
    iter=iter+1;
    f=log(h_theta(theta,X))*y'+log(1-h_theta(theta,X))*(1-y)';
    f=(-f);
    f=f+lambda*sum(theta.^2);
    gap=abs(f-oldf);
    if gap<0.01
        break;
    else
        oldf=f;
    end
    Xt=h_theta(theta,X)-y;
    g=X*Xt';
    g=g+lambda*[theta(1:end-1);0];
    theta=theta-eta*g;
    if mod(i,1000)==0
        eta=eta*0.8;
%         fprintf('iter=%d cost=%d\n',iter,f);
    end
end
% fprintf('iter=%d cost=%d\n',iter,f);
w=theta;
end

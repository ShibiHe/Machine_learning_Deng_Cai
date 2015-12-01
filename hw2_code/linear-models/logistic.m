function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
y(y==-1)=0;
theta=randn(size(X,1)+1,1);
X=[ones(1,size(X,2));X];
eta=1;
iter=0;
oldf=100000;
for i=1:1000000
    iter=iter+1;
    f=log(h_theta(theta,X))*y'+log(1-h_theta(theta,X))*(1-y)';
    f=-f;
    if abs(f-oldf)<0.01
        break;
    else
        oldf=f;
    end
    Xt=h_theta(theta,X)-y;
    g=X*Xt';
    theta=theta-eta*g;
    if mod(i,1000)==0
        eta=eta*0.8;
%         fprintf('iter=%d cost=%d\n',iter,f);
    end
end
w=theta;
end

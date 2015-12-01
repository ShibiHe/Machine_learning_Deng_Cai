function answer = h_theta(theta, X)
    %1/(1+realmin*10^304)=0.9998
    sigmoid_min=realmin*10^304;
    X=theta'*X;
    exp_mX=exp(-X);
    exp_mX(isinf(exp_mX))=realmax;
    exp_mX(exp_mX<sigmoid_min)=sigmoid_min;
    X=1./(1+exp_mX);
    answer = X;
end
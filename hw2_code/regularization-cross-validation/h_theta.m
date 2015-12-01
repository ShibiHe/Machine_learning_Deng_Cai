function answer = h_theta(theta, X)
    %1/(1+realmin*10^304)=0.9998
    sigmoid_min=realmin*10^304;
    X=theta'*X;
    exp_mX=exp(-X);
    exp_mX(isinf(exp_mX))=realmax;
    exp_mX(exp_mX<sigmoid_min)=sigmoid_min;
% the next codes are too fucking slow!!
%     for i=1:length(exp_mX)
%         if isinf(exp_mX(i))
%             exp_mX(i)=realmax;
%         end
%         if exp_mX(i)<sigmoid_min
%             exp_mX(i)=sigmoid_min;
%         end
%     end
    X=1./(1+exp_mX);
    answer = X;
end
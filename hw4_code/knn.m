function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%

% YOUR CODE HERE
D=pdist2(X',X_train');
[B,I]=sort(D,2);
I=I(:,1:K);
y=zeros(1,size(X,2));
I=reshape(y_train(I(:)),size(I));
classes=max(y_train)+1;
for i=1:size(X,2)
    label=zeros(1,classes);
    for j=1:K
        label(I(i,j)+1)=label(I(i,j)+1)+1;
    end
    [~,I2]=max(label);
    y(i)=I2-1;
end
end


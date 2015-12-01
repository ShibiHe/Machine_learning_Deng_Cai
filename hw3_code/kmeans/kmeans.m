function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE
[N,P]=size(X);
indices=ceil(N*rand(K,1));
while (sum(unique(indices)~=indices)==0)
    indices=ceil(N*rand(K,1));
end

ctrs=X(indices,:);
idx=ones(1,N);
old_idx=zeros(1,N);
iter_ctrs(:,:,1)=ctrs;
iter=1;
while (sum(idx~=old_idx)>0)
    old_idx=idx;
    D=pdist2(X,ctrs);
    iter=iter+1;
    [~,I]=min(D,[],2);
    idx=I';
    for k=1:K
        ctrs(k,:)=mean(X(I==k,:));
    end
    iter_ctrs(:,:,iter)=ctrs;
end
iter=iter-1;
end

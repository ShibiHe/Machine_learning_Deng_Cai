function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE

[N,~]=size(X);
W=zeros(N,N);
D=pdist2(X,X);
[dis,index]=sort(D,2);

K_neighbors=dis(:,1:k);
K_neighbors(K_neighbors>threshold)=0;
K_neighbors(K_neighbors>0)=1;
for i=1:N
    W(i,index(i,1:k))=K_neighbors(i,:);
    W(index(i,1:k),i)=K_neighbors(i,:)';
end
end

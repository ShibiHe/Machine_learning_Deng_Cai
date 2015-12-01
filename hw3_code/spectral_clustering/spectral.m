function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE
N=size(W,1);
D=diag(sum(W));
D=D+0.001*eye(N);

[V,d]=eigs(D-W,D,k,'SM');
idx=kmeans(V,k);

end

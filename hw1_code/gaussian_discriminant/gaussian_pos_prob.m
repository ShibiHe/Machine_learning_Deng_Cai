function p = gaussian_pos_prob(X, Mu, Sigma, Phi)
%GAUSSIAN_POS_PROB Posterior probability of GDA.
%   p = GAUSSIAN_POS_PROB(X, Mu, Sigma) compute the posterior probability
%   of given N data points X using Gaussian Discriminant Analysis where the
%   K gaussian distributions are specified by Mu, Sigma and Phi.
%
%   Inputs:
%       'X'     - M-by-N matrix, N data points of dimension M.
%       'Mu'    - M-by-K matrix, mean of K Gaussian distributions.
%       'Sigma' - M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of
%                   K Gaussian distributions.
%       'Phi'   - 1-by-K matrix, prior of K Gaussian distributions.
%
%   Outputs:
%       'p'     - N-by-K matrix, posterior probability of N data points
%                   with in K Gaussian distributions.

N = size(X, 2);
K = length(Phi);
p = zeros(N, K);

% Your code HERE
joint = zeros(N, K);
for i =1:N
    for k = 1:K
        joint(i,k) = Phi(k)/sqrt(det(Sigma(:,:,k)))*exp(-0.5*(X(:,i) - Mu(:,k))'*inv(Sigma(:,:,k))*(X(:,i) - Mu(:,k)));
    end
    p(i,:) = joint(i,:)/sum(joint(i,:));
end




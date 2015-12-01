function p = feedforward(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%
%   Input:  Theta1 -- weights between input-hidden layers, 401x25 matrix
%           Theta2 -- weights between hidden-output layers, 26x10 matrix
%                X -- test set, 400xP matrix, P is size of testing set
%
%   Output: p -- predicted labels, 1xP row vector

% Note:
% The matrix X contains the examples in columns.
% The matrices Theta1 and Theta2 contain the parameters for each unit in
% column. Specifically, the first column of Theta1 corresponds to the first
% hidden unit in the second layer.

% YOUR CODE HERE
P=size(X,2);
X=[ones(1,P);X];
a2=h_theta(Theta1,X);
a2=[ones(1,P);a2];
a3=h_theta(Theta2,a2);
[M,I]=max(a3);
p=I;
end

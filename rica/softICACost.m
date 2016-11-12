%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
z1 = W*x;
z2 = W'*z1;
z3 = z2-x;
a3 = z3.^2;
cost = (0.5*sum(a3(:))+params.lambda*sum(sum(sqrt(z1.^2+params.epsilon))))/size(a3,2);
Wgrad = params.lambda*z1./(sqrt(z1.^2+params.epsilon))*x'+ W*2*z3*x'+2*z1*z3';
Wgrad= Wgrad./size(a3,2);
% unproject gradient for minFunc

grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);

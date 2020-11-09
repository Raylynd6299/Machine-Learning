function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% With F = Number of features
%      M = Number of training examples
% Size of X = M x (F + 1)
% Size of Y = M x 1
% Size of Theta = (F + 1) * 1 


% The Result of this operation is a matrix of size : M x 1 
predictions = sigmoid(X * theta);

J_N_Reg = ( -1 / m) * sum ( ( y .* log(predictions) ) + ( ( 1 - y ) .* log( ( 1 - predictions ) ) ) );

grad_no_Reg = ( ( 1 / m ) * ( X' * ( predictions - y ) ) );

%Regularization Cost function

J = J_N_Reg + ( ( lambda / (2 * m) ) * sum( theta(2:length(theta)) .^ 2 ) );

%Regularization gradient

grad_aux = grad_no_Reg(1);

grad_Reg = grad_no_Reg + ( ( lambda / m ) * theta );

grad = [grad_aux ; grad_Reg(2:length(theta))];

% =============================================================

grad = grad(:);

end

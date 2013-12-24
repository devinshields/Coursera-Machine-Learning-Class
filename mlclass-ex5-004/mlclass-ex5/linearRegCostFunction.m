function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

LOG('---------- running linearRegCostFunction -------------------')

% ====================== helpful logging ======================
LOG('size(X) == %s', mat2str(size(X)))
LOG('size(y) == %s', mat2str(size(y)))
LOG('size(theta) == %s', mat2str(size(theta)))
LOG("\n\n")


h_y = X*theta - y;
J = (1/(2*m)) * (h_y' * h_y) + (lambda / (2*m)) * (theta(2:end)' * theta(2:end));


grad = (1/m) * (X' * h_y) + (lambda/m) * (theta .* [0; ones(length(theta)-1, 1)]);




LOG('---------- finished linearRegCostFunction -------------------')

% =========================================================================

grad = grad(:);

end

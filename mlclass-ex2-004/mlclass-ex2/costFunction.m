function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

% 'Log: running cost function'


% non-vectorized cost function (for testing/a first go)
J = 0;
for i = 1:m
  h = sigmoid(X(i, :) * theta);
  J += (-1 * y(i) * log(h))  - ((1 - y(i)) * log(1 - h));
end
J /= m;


% non-vectorized gradient function
for j = 1:length(theta)
  for i = 1:m
    h        = sigmoid(X(i, :) * theta);
    grad(j) += (h - y(i)) * X(i, j);
  end
  grad(j) /= m;
end





% =============================================================

end

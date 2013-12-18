function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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




% non-vectorized cost function (for testing/a first go)
J = 0;
for i = 1:m
  h = sigmoid(X(i, :) * theta);
  J += (-1 * y(i) * log(h))  - ((1 - y(i)) * log(1 - h));
end
J /= m;
% add the regularization term
J += lambda / (2*m) * (theta(2:end)' * theta(2:end));



% non-vectorized gradient function
for j = 1:length(theta)
  for i = 1:m
    h        = sigmoid(X(i, :) * theta);
    grad(j) += (h - y(i)) * X(i, j);
  end
  grad(j) /= m;
  % add the regularization term, if it's not the intercept (j > 1)
  grad(j) += lambda / m * theta(j) * (j > 1);
end

% =============================================================

end

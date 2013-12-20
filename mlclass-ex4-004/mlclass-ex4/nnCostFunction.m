function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%


LOG('--------------- running nnCostFunction ---------------')


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== helpful logging ======================
LOG('size(X) == %s', mat2str(size(X)))
LOG('size(y) == %s', mat2str(size(y)))
LOG('size(Theta1) == %s', mat2str(size(Theta1)))
LOG('size(Theta2) == %s', mat2str(size(Theta2)))
LOG('size(nn_params) == %s', mat2str(size(nn_params)))


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%  forward propogating features through the network
A2 = sigmoid([ones(m, 1) X]           * Theta1');
A3 = sigmoid([ones(size(A2, 1), 1) A2] * Theta2');


% map the vector y (in range(1, num_clases)) into a boolean membership matrix, Y.
%     borrowed from: https://gist.github.com/denzilc/1360709
Y = eye(num_labels)(y,:);


% get element-wise errors, then reduce-agg the whole matrix
D1 = -Y    .* log(A3);
D2 = (1-Y) .* log(1-A3);
s0 = (1/m) * sum((D1 - D2)(:));


% regularization cost - ignore bias terms in each firest column
reg_params = [Theta1(:,2:end)(:) ; Theta2(:,2:end)(:)];
s1         = (lambda / 2 / m) * (reg_params' * reg_params);

% sum to get the final penalty
J = s0 + s1;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% back propagate classifications errors for each example in the training set
X0 = [ones(m, 1) X];

for t = 1:m
  xt  = X0(t, :);
  yt  = Y(t, :);
    
  %  forward propogate
  a1 = xt;
  a2 = sigmoid(  a1    * Theta1');
  a3 = sigmoid([1, a2] * Theta2');

  % back propogate
  d3 = a3 - yt;
  d2 = Theta2' * d3 .* 
  keyboard

end





% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


LOG('--------------- nnCostFunction complete ---------------')

end

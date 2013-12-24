function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
LOG("\n\n")


% ====================== YOUR CODE HERE ======================
% Part 1: Feedforward the network.

% prepend a bias column, transform y into a binary class membership matrix
X = [ones(m, 1) X];
y = eye(num_labels)(y,:);
 

% forward propogate, cache activation levels for the backprop step
a1 = X;                         % layer 1
z2 = a1 * Theta1';

a2 = sigmoid(z2);               % layer 2
a2 = [ones(size(a2, 1),1) a2];  %   append bias

z3 = a2 * Theta2';
a3 = sigmoid(z3);               % layer 3
 

% ====================== log forward propogation ======================
LOG("------------------------------------------")
LOG('size(a1) == %s', mat2str(size(a1)))
LOG('size(z2) == %s', mat2str(size(z2)))
LOG('size(a2) == %s', mat2str(size(a1)))
LOG('size(z3) == %s', mat2str(size(z2)))
LOG('size(a3) == %s', mat2str(size(a3)))
LOG("\n\n")


% ====================== calculate the error cost ======================
% get element-wise errors, then reduce-agg the whole matrix to a scalar
c0 = (1/m) * sum(((-y .* log(a3)) - (1-y) .* log(1-a3))(:));

% add regularization cost - ignore bias terms in each firest column
reg_params = [Theta1(:,2:end)(:) ; Theta2(:,2:end)(:)];
c1         = (lambda / 2 / m) * (reg_params' * reg_params);

J = c0 + c1;


% =========== back propogate errors, one node layer at a time =================
d3 = a3 - y;                                            % output node errors
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);     % hidden node errors, ignore the bias parameter

% the desired diff (delta cap) is a weighted average
%     of the n-1 layer's activation (a~)
%     and the change we want to induce (d+)
delta_cap2 = d3' * a2; 
delta_cap1 = d2' * a1;

% get the gradient, but scaled on
%       learning rate (lambda)
%       sample size (m)
Theta1_grad       = ((1/m) * delta_cap1) + ((lambda/m) * (Theta1));
Theta1_grad(:,1) -= ((lambda/m) * (Theta1(:,1)));                     % remove the regularization term from bias nodes

Theta2_grad       = ((1/m) * delta_cap2) + ((lambda/m) * (Theta2));
Theta2_grad(:,1) -= ((lambda/m) * (Theta2(:,1)));                     % remove the regularization term from bias nodes



% ====================== logging ======================
LOG("------------------------------------------")
LOG('size(d3) == %s', mat2str(size(d3)))
LOG('size(d2) == %s', mat2str(size(d2)))
LOG('size(delta_cap2) == %s', mat2str(size(delta_cap2)))
LOG('size(delta_cap1) == %s', mat2str(size(delta_cap1)))
LOG('size(Theta1_grad) == %s', mat2str(size(Theta1_grad)))
LOG('size(Theta2_grad) == %s', mat2str(size(Theta2_grad)))
LOG("\n\n")




% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


LOG('--------------- nnCostFunction complete ---------------')

end

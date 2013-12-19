function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% NOTES:
%         - 400 input parameters, manually adding 1 bias parameter
%         - they feed into 25 hidden layer nodes
%         - 25 hidden nodes and 1 bias node feed into 10 output nodes.
%         - the prediction from the neural network will be the label that has the largest output (hθ(x))k.


% append a bias column, forward propagation via matrix multiplication
X1 = sigmoid([ones(m, 1) X]  * Theta1');
X2 = sigmoid([ones(m, 1) X1] * Theta2');


% map maximum value predictions/indicies into classifications
[x, p] = max(X2, [], 2);


% =========================================================================
end

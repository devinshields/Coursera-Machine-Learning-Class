function [all_theta] = oneVsAll(X, y, num_labels, lambda)

  m = size(X, 1); % sample size
  n = size(X, 2); % degrees of freedom +1. or bobanparameter count

  % prepend an intercept term column to the X data matrix
  X = [ones(m, 1) X];

  % init a blank multi-model paramter matrix. Will eventually get output.
  all_theta = zeros(num_labels, n + 1);



  % ============= Serious Work Gets Done Here Guys ======================
  %keyboard % !!!
  
  warning('off', 'Octave:possible-matlab-short-circuit-operator');

  for class_num = 1:num_labels
    [theta] = fmincg (@(t)(lrCostFunction(t, X, y == class_num, lambda)), zeros(n + 1, 1), optimset('GradObj', 'on', 'MaxIter', 50));

    all_theta(class_num, :) = theta';
  end

  



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%












% =========================================================================


end

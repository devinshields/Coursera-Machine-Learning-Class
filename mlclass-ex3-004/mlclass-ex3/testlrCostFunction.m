function testlrCostFunction()

  rand ("seed", 42);

  % set some sim paramters
  n     = 15;

  lambda = .00234;
  theta = [-.2, .1, .3]';
  

  % generate random feature and class data
  X = [ones(n,1), rand(n, length(theta) - 1 )];
  y = rand(n, 1) > .5;



  % run some tests
  [J, grad] = lrCostFunction(theta, X, y, lambda)
  disp(sprintf('\n------------------------------------\n'))


  [J, grad] = lrCostFunction1(theta, X, y, lambda)
  disp(sprintf('\n------------------------------------\n'))

  [J, grad] = lrCostFunction2(theta, X, y, lambda)
  disp(sprintf('\n------------------------------------\n'))





end

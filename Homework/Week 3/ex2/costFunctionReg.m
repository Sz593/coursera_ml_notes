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


h = sigmoid(X * theta);
% J = (1/m) * (-y' * log(h) - (1-y)' * log(1-h));
grad = (1/m) * X' * (h - y);

%size(theta)     % 28, 1
%size(X)         % 118, 28
%size(y)         % 118, 1
%size(lambda)    % 1, 1
%size(h)         % 118, 1

n = length(theta);

term1 = 0;
for i = 1:m,
  calc_sum = -y(i, 1) * log(h(i, 1)) - (1 - y(i, 1)) * log(1 - h(i, 1));
  term1 = term1 + calc_sum;
 end;
 
term_reg = 0;
for j = 2:n,
  calc_reg_sum = theta(j, 1) ^ 2;
  term_reg = term_reg + calc_reg_sum;
 end;

J = (1 / m) * term1 + (lambda / (2 * m)) * term_reg;







for j = 2:n,
  reg_term = (lambda / m) * theta(j, 1);
  
  sum_term = 0;
  for i = 1:m,
    calc_term = (h(i, 1) - y(i, 1)) * X(i, j);
    sum_term = sum_term + calc_term;
   end;
  
  grad(j, 1) = (1 / m) * sum_term + reg_term;
 end;

% =============================================================

end

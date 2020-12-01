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

%temp = - y.*log(h_thetax) -(ones(m,1)-y).*log(1-h_thetax); 答案这里还是点乘，我是直接矩阵相乘了
%J = sum(temp)/m;
%temp = theta;
%temp(1) = 0;
%J = J + 0.5*lambda*(temp'*temp)/m;
%
%grad = X'*(h_thetax -y)/m;
%temp = theta;
%temp(1) = 0;
%temp = temp*lambda/m;
%grad = grad + temp; 

h_thetax = sigmoid(X*theta);
J = -(y'*log(h_thetax)+(ones(size(y))-y)'*log(1-h_thetax))/m+lambda*(theta'*theta-theta(1)^2)*0.5/m

grad = ((h_thetax-y)'*X)'/m
temp = theta;
temp(1) = 0;
temp = temp*lambda/m;
grad = grad + temp; 
% =============================================================

grad = grad(:); %这句有什么意义？！

end

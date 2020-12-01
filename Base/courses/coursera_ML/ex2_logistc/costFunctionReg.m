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
n = size(theta);

for i = 1:m
   h_thetax = sigmoid(X(i,:)*theta);
   J = J - y(i)*log(h_thetax) -(1-y(i))*log(1-h_thetax);
end
for i = 2:n
    J = J + 0.5*lambda*theta(i)*theta(i);
end
  J = J/m;

  
for j =1:n
    sum = 0;
    for i = 1:m
      h_thetax = sigmoid(X(i,:)*theta);
      sum = sum + (h_thetax - y(i))*X(i,j); 
    end 
    if(j==1)
       grad(j) = sum/m;
    else
        grad(j) = theta(j)*lambda/m + sum/m;
    end
end



% =============================================================

end

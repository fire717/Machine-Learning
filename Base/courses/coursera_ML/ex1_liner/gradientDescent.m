function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    tmp = theta       %simultaneously update
    tmp(1) = theta(1)-alpha*(1/m)*sum(X*theta-y)
    tmp(2) = theta(2)-alpha*(1/m)*((X*theta-y)'*X(:,2))
    theta = tmp

%    n = length(theta);
%    theta1  = theta; 
%    for i = 1:n
%      S  =  0;
%      for j = 1:m
%        S  =  S + (X(j,:)*theta-y(j)).*X(j,i);
%      end
%      S = S*alpha/m;
%      theta1(i) = theta(i) - S;
%    end
%     theta = theta1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

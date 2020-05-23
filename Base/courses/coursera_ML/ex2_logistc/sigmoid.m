function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


%for i=1:size(z),
%  g(i)=1/(1+exp(-z(i)));
%end;  因为测试不是列向量是行向量，所以只有第一位对。所以还是要按答案写考虑矩阵
%
[m,n] = size(z);

for i = 1:m
    for j= 1:n
       g(i,j) = 1/(1+exp(-z(i,j)));        
    end
end

% =============================================================

end

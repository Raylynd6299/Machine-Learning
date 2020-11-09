function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

[m,n] = size (z);
for fila = 1:m
    for columna = 1:n
        g(fila,columna) = (1/(1+(e^(-z(fila,columna)))));
    end
end

% =============================================================

end

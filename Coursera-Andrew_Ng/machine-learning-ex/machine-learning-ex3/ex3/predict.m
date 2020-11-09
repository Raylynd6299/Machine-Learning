function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%Agregue el valor del bias unit 1's a la primer caracteristica
% 5000 x 400
X = [ones(size(X,1), 1) X];
% 5000 x 401

%thetha1 supone que biene de n numeros de entradas pero contaba con el numero de columns correcto contemplando la bias unit
% 25 x 401
%La nota del profesor marcaba que esto fue armado con la primer fila como la inicada para el bias unit  de a2

A2 = sigmoid(Theta1 * X');

A2 = [ ones(1,size(A2,2)); A2 ];

A3 = sigmoid(Theta2 * A2);

size(A3)

% This is de same 
%porque en esta matris los traning examples estan como vectores fila
%[x,p] = max(A3);

[x,p] = max (A3', [] , 2);




% =========================================================================


end

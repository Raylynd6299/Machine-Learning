function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);                                                 %Number of examples
num_labels = size(all_theta, 1);                                %Number of labels

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);                                       %Prediction for each example

% Add ones to the X data matrix

X = [ones(m, 1) X]; 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% =========================== MY CODE =====================================

% The size of X is m x (F + 1) when F is equal to the number of features 
% The size of all_theta is num_labels x (F + 1)

% thus X * all_theta'  will give me as a result de Z number tu sigmoid function
% the sigmoid function work for a matrix, well the result of them is the probability of be the label of this column

%To obtain all prediction 

probabilitys = sigmoid( X * all_theta' );
% To obtain the max probability for each training example 
[x,p] = max(probabilitys,[],2); % the third parameter is the dimension to obtain the max value

% x is the value to the probaility higher to the label in the same row in the p matrix
% p is the matrix that have the laber obtanied to have the good forecast
% =========================================================================


end

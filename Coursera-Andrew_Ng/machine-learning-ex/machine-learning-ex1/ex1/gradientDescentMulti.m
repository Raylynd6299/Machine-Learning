function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
J = length(theta);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    theta_up=zeros(J,1);

    for iter_2 = 1 : J

        sum = 0;
        for iter3=1:m
            sum = sum + ( (theta' * X(iter3,:)') - y(iter3) ) * ( ( X( iter3 , : )' )( iter_2 ) ) ;
        end

        theta_up(iter_2) = theta(iter_2) - ( alpha * (1/m) * sum );

    end
    %fprintf('Nuevas Thethas %f\n', theta_up);

    theta = theta_up;   


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

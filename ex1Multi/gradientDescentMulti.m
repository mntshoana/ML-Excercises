function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
% Gradient descent and updates theta using the learning rate alpha

    % Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);
    for iter = 1:num_iters
        c = 1/m;
        h = X * theta;
        DeltaJ =  c * X' * (h - y);
        theta = theta - alpha * DeltaJ;

        % Save the cost J in every iteration
        J_history(iter) = computeCostMulti(X, y, theta);
    end
end

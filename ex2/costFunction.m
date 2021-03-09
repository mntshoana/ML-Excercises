function [J, grad] = costFunction(theta, X, y)
% Compute cost and gradient for logistic regression

    m = length(y); % number of training examples
    k = 1/m;
    
    % HYPOTHESIS
    z = X * theta;
    h = sigmoid(z);

    % COST
    log1 = log(h);
    log0 = log(1 - h);
    a = - y' * log1;
    b = (1 - y)' * log0;
    J = k * (a - b);

    % GRADIENT
    d = X' * ( h - y );
    grad = k * d;

end

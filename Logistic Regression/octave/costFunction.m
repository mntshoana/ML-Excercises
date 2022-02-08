function [J, deltaJ] = costFunction(theta, X, y)
%   Logistic regression
%   Cost and gradient 
    m = length(y);
    
    % HYPOTHESIS
    z = X * theta;
    %FILE: sigmoid.m
    h = sigmoid(z);

    % COST
    a = - y' * log(h);
    b = (1 - y)' * log(1 - h);
    J = 1/m * (a - b);

    % GRADIENT
    deltaJ = 1/m * X' * ( h - y );

end

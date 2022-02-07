function [J, grad] = costFunction(theta, X, y)
%   Logistic regression
%   Cost and gradient 
    m = length(y);
    
    % HYPOTHESIS
    z = X * theta;
    %FILE: sigmoid.m
    h = sigmoid(z);

    % COST
    log1 = log(h);
    log0 = log(1 - h);
    a = - y' * log1;
    b = (1 - y)' * log0;
    J = 1/m * (a - b);



end

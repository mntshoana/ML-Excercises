function [J, grad] = costFunction(theta, X, y)
%   Logistic regression
%   Cost and gradient 
    m = length(y);
    
    % HYPOTHESIS
    z = X * theta;
    %FILE: sigmoid.m
    h = sigmoid(z);



end

function [J, regDeltaJ] = costFunctionReg(X, y, theta, lambda)
% Computes cost and the gradient with regularization (not the gradient decent)
%   J = COSTFUNCTIONREG(X, y, theta, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression
    [J, deltaJ] = costFunction(X, y, theta);

    m = length(y);
    alpha = lambda / ( 2 * m );
    % Do not regularize initial theta (theta0) 
    _theta = theta( 2 : end ); 
    summation = sum( _theta .^ 2 );
    reg = alpha * summation;
    J = J + reg; % Regularized cost
    
    % Remember, DO NOT regularize initial theta (theta0) 
    regDeltaJ = (lambda / m) * _theta;
    regDeltaJ = [0; regDeltaJ];
    regDeltaJ = deltaJ + regDeltaJ;
end
function [theta] = normalEqn(X, y)
%  Computes linear regression using the normal equations.
%  Normal Equation is an analytical approach to Linear Regression with a Least Square Cost Function
    theta = zeros(size(X, 2), 1);

    a = pinv(X' * X);
    b = X';
    theta = a * b * y ;

end

function [theta] = normalEqn(X, y)
%  Computes linear regression using the normal equations.
%  Normal Equation is an analytical approach to Linear Regression with a Least Square Cost Function
    colSize = size(X, 2);
    theta = zeros(colSize, 1);

    a = pinv(X' * X);
    b = X';
    theta = a * b * y ;

end

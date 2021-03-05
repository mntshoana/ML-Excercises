function [theta] = normalEqn(X, y)
%  Computes the closed-form solution to linear regression using the normal equations.

    theta = zeros(size(X, 2), 1);

    a = pinv(X' * X);
    b = X';
    theta = a * b * y ;

end

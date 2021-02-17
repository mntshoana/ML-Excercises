function J = computeCost(X, y, theta)
% Cost for linear regression
    h = X * theta;
    a = (h - y)';
    b = h - y ;

    m = length(y); % number of training examples
    mean = 1 / (2 * m);

    J = mean * a * b;

end

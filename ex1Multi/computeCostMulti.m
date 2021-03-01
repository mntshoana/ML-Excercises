function J = computeCostMulti(X, y, theta)
% Linear regression cost (with multiple variables)

    % Initialize useful values
    m = length(y);
    mean = 1 / (2 * m);
    h = X * theta;

    a = (h - y)';
    b = h - y ;
    J = mean * a * b;
end

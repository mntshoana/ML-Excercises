function p = predict(theta, X)
% Predict label ( 0 or 1 ) using logistic regression parameters theta and a threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    m = size(X, 1); % Number of training examples
    p = zeros(m, 1);
    prob =  sigmoid(X * theta);

    cond1 = prob >= 0.5;
    cond2 = prob < 0.5;
    p = 1 .*cond1 + 0 .*(cond2) + prob .* (~(cond1 | cond2));

end

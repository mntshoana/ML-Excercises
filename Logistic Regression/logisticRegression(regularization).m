%% Logistic Regression with regularization
%
% INITIALIZE
clear;
close all;
clc;

data = load('data/ex2data2.txt');
X = data(:, [1, 2]);
y = data(:, 3);

addpath("octave")
% ====================   Plotting   ====================
plotData(X, y);
hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% ========== Regularized Logistic Regression ===========
% Add Polynomial Features
% FILE: polynomial.m
X = polynomial(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% regularization parameter
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost, deltaJ] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', deltaJ(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, deltaJ] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', deltaJ(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
% ========== Effect of regularization ===========
m = length(y); % number of training examples
num_iters = 20000;
alpha = 3;

fprintf('\nShowing gradient decent with lambda = 1\n');
J_history = zeros(num_iters, 1);
theta = zeros(size(X,2), 1);
lambda = 1;

for iter = 1:num_iters
    [cost, deltaJ] = costFunctionReg(theta, X, y, lambda);
    J_history(iter) = cost;
    theta = theta - alpha * deltaJ;
end

%FILE: plotDecisionBoundary.m
plotDecisionBoundary(theta, X, y);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nShowing gradient decent with lambda = 0\n');
J_history = zeros(num_iters, 1);
theta = zeros(size(X,2), 1);
lambda = 0;

for iter = 1:num_iters
    [cost, deltaJ] = costFunctionReg(theta, X, y, lambda);
    J_history(iter) = cost;
    theta = theta - alpha * deltaJ;
end

plotDecisionBoundary(theta, X, y);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nShowing gradient decent with lambda = 20\n');
J_history = zeros(num_iters, 1);
theta = zeros(size(X,2), 1);
lambda = 20;

for iter = 1:num_iters
    [cost, deltaJ] = costFunctionReg(theta, X, y, lambda);
    J_history(iter) = cost;
    theta = theta - alpha * deltaJ;
end

plotDecisionBoundary(theta, X, y);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
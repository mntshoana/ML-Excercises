%% Machine Learning - Exercise 2: Logistic Regression
%
% INITIALIZE
clear;
close all;
clc;

data = load('data/ex2data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);

addpath("octave")
% ====================   Part 1: Plotting   ====================
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

%FILE: plotData.m
plotData(X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
fprintf('Paused. Press enter to continue.\n\n');
pause;


% ============== Part 2: Compute Cost and Gradient ==============
[m, n] = size(X);
X = [ones(m, 1), X];

%FILE: costFunction.m
% Compute and display initial cost and gradient
initial_theta = zeros(n + 1, 1);
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

% Compute and display cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);
fprintf('\nCost at test theta: %f\n', cost);
fprintf('Expected cost (approx): 0.218\n');
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Optimizing using fminunc  =============
% Create options structure for optimization function
% In this case, this function is fminunc
options = optimset('GradObj', 'on', % Second arg returned will be the first derivative of gradient decent
                     'MaxIter', 400);

%  Optimization function fminun tries to determine the local minimum of a given function (in this case, a cost function)
[theta, cost] = fminunc( @(t)( costFunction(t, X, y) ),
                        initial_theta, options );

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('Expected cost (approx): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');
% TODO

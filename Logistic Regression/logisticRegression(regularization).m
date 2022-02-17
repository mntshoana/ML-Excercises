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
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
%TODO
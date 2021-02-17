%% Machine Learning - Exercise 1: Linear Regression
% x is the population size in 10,000s
% y is the profit in $10,000s
%
% INITIALIZE
clear;
close all;
clc;
% ==================== Part 1: Basic Function ====================
fprintf('Basic Function\n');
fprintf(' - A 5x5 Identity Matrix: \n');
% FILE: warmUpExercise.m
warmUpExercise()
fprintf('Program paused. Press enter to continue.\n');
pause;
%% ===================     Part 2: Plotting   ====================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
% FILE: plotData.m
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============== Part 3: Cost and Gradient descent =============
X = [ones(m, 1), data(:,1)]; % M X 2 matrix
theta = zeros(2, 1); % fitting parameters
fprintf('\nTesting the cost function ...\n')
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');


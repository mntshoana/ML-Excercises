%% Machine Learning - Exercise 1: Linear Regression
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
fprintf('Paused. Press enter to continue.\n\n');
pause;

%% ===================     Part 2: Plotting   ===================
% x= population size in 10,000s
% y= profit in $10,000s
fprintf('Plotting Data\n')
data = load('../ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
% FILE: plotData.m
plotData(X, y);
fprintf('Paused. Press enter to continue.\n\n');
pause;

%% ============== Part 3: Cost and Gradient descent =============
fprintf('Testing the cost function ...\n')
X = [ones(m, 1), data(:,1)];  % MX2
theta = zeros(2, 1);          % 2X1
J = computeCost(X, y, theta);
fprintf(' - With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf(' - Expected cost value (approx) 32.07\n');
J = computeCost(X, y, [-1 ; 2]);
fprintf('\n - With theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf(' - Expected cost value (approx) 54.24\n');
fprintf('Paused. Press enter to continue.\n\n');
pause;

fprintf('Running Gradient Descent ...\n')
iterations = 1500;
alpha = 0.01;
[theta,Jhist] = gradientDescent(X, y, theta, alpha, iterations);
fprintf('- Theta found:\n');
fprintf('  %f\n', theta);
fprintf('- Expected theta values (approx)\n');
fprintf('  -3.6303\n  1.1664\n\n');

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%Predict
predict1 = [1, 3.5] *theta; % population of 35,000
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta; % population of 70,000
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Paused. Press enter to continue.\n\n');
pause;

%% =========== Part 4: Visualizing J(theta_0, theta_1) ==========
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
      t = [theta0_vals(i); theta1_vals(j)];
      J_vals(i,j) = computeCost(X, y, t);
    end
end

% With the surf command, we'll need to transpose J_vals
% or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
printf('Paused. Press enter to continue.\n\n');
pause;

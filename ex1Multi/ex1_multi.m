%% Machine Learning Exercise 1: Linear regression with multiple variables
%
% INITIALIZE
clear;
close all;
clc;
% ================ Part 1: Feature Normalization ================
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[Xnorm mu sigma] = featureNormalize(X);

% Add intercept term to X
Xnorm = [ones(m, 1) Xnorm];
fprintf('Paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(Xnorm, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
% Cost
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Theta after gradient descent
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = [1, (1650 - mu)/sigma , (3 - mu)/sigma] *theta;% You should change this
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================
fprintf("Using normal equation...\n");
X = [ones(m, 0), X]

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
price = [1, 1650  , 3 ] *theta; % You should change this
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

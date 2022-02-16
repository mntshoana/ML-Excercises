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
%TODO
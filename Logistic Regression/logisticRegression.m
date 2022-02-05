%% Machine Learning - Exercise 2: Logistic Regression
%
% INITIALIZE
clear;
close all;
clc;

data = load('data/ex2data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);
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
% TODO
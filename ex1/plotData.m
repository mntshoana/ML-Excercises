function plotData(x, y)
%   Plots data points
%   x = population and y = profit.
    figure; % open a new figure window
    plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data
    ylabel('Profit in $10,000s');
    xlabel('Population of City in 10,000s');
end

function plotDecisionBoundary(theta, X, y)
% Plots X and y with the decision boundary defined by theta
%   X is an MxN matrix, N>=3 , with the first column all-ones
    
    plotData(X(:,2:3), y);
    hold on

    if size(X, 2) <= 3
        % Need 2 points to define a line, so choose two endpoints
        plot_x = [ min(X(:,2)) -2,  max(X(:,2)) +2];
        % Calculate the decision boundary line
        plot_y = (-1. / theta(3)) .* (theta(2) .* plot_x + theta(1));

        plot(plot_x, plot_y)
        legend('Admitted', 'Not admitted', 'Decision Boundary')
        axis([30, 100, 30, 100])
    else
        % Here is the grid range
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);

        z = zeros(length(u), length(v)); %grid
        for i = 1:length(u)
            for j = 1:length(v)
                %file polynomial.m
                z(i,j) = polynomial( u(i), v(j) ) * theta;
            end
        end
        z = z'; % transpose before calling contour

        % Plot z = 0
        % Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2)
    end
    hold off

end

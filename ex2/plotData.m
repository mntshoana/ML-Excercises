function plotData(X, y)
% Plots the data points X and y
%    X is assumed to be a Mx2 matrix.

    figure;
    hold on;
    pos = find( y == 1 );
    neg = find( y == 0 );

    plot( X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    plot( X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);
    hold off;

end

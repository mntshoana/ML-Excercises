function [X_norm, mu, sigma] = featureNormalize(X)
  % Returns a normalized version where the mean
  % value of each feature is 0 and the standard deviation is 1.
mu = mean(X);
sigma = std(X);
a = X - mu;
X_norm = a./ sigma;

end

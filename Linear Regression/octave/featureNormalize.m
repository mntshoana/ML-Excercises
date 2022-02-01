function [X_norm, mu, sigma] = featureNormalize(X)
  % Returns a normalized version where the mean
  % value of each feature is 0 and the standard deviation is 1.
mu = mean(X);
printf("mu: %.f, %.f\n",mu)
sigma = std(X);
printf("sigma: %.f, %.f\n",sigma)
a = X - mu;
printf("a: %.f\n",a(1:3,:))
X_norm = a./ sigma;
printf("X_norm: %.f, %.f\n",X_norm(1:3,:))

end

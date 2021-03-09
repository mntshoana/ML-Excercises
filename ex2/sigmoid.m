function sig = sigmoid(z)
%Computes sigmoid function

    epsilon = exp(-z);
    den = 1 + epsilon;
    sig = 1./den;

end

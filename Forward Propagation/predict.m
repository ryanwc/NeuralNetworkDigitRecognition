function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% initialize useful values
m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% initialize hidden layer
hiddenOne = zeros( size(X, 2), size(Theta2, 1) );

% initialize final output
MaxFinal = zeros(size(X, 1), 1);
% need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% return index of max in row to know what hidden layer is
hiddenOne = sigmoid(X * Theta1');
% add bias unit to hidden layer
hiddenOne = [ones(size(hiddenOne, 1), 1) hiddenOne];

% return index of max in row to know what the prediction is
[MaxFinal, p] = max( sigmoid(hiddenOne * Theta2'), [], 2 );

% =========================================================================


end

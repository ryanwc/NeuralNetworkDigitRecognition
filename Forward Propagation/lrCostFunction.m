function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% need to return the following variables correctly 
% J = cost;
% grad = gradient vector;

% Initialize some useful values
m = length(y); % number of training examples

predictions = sigmoid(X*theta); % predictions of hypothesis on all m examples
hypoError = predictions-y; % error of each prediction

% costOfError = cost of how far away the hypo is from the classification
costOfError = ( -y .* log(predictions) ) - ( (1 - y) .* log(1-predictions) );

%%% withOUT regularization %%%

noRegCost = (1/m) * sum(costOfError);  % vectorized logistic regression cost function
grad = (1/m) * (X' * hypoError);  % vectorized logistic regression gradient

%%% WITH regularization %%%

squaredParams = theta.^2;  % sqaures of the params
squaredParams(1) = 0;  % do not regularize the intercept param
costRegTerm = ( lambda / (2*m) ) * sum(squaredParams);

J = noRegCost + costRegTerm; % regularized cost function; return this

gradRegTerms = (lambda / m) * theta;  % calc reg term for all grads
gradRegTerms(1) = 0;  % do not regularize the intercept param

grad = grad + gradRegTerms;

% =============================================================

%grad = grad(:);

end

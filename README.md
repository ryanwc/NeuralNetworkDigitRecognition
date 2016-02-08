# Overview

Neural Networks: Digit Recognition is a series of MATLAB scripts that use logistic regression and neural networks to classify handwritten digits.

The project is divided into two parts: 1) comparing one-vs-all classification by logistic regression with one-vs-all classification by a neural network that uses given parameters (i.e., implementing forward propagation through the neural network with parameters known to perform well on the dataset) and 2) one-vs-all classification by a neural network that uses a backpropagation algorithm to learn parameters.  There are folders in this repo that correspond to each part.

This project was completed for Stanford University's [Machine Learning](https://www.coursera.org/learn/machine-learning/) course offered on Coursera.

# Technical Requirements

To run this software, you need Octave or MATLAB.  MATLAB is proprietary, and Octave is open source and mostly compatible with MATLAB.  Both are designed for complex numerical calculations.

You can download Octave [here](https://gnu.org/software/octave/).  However, I wrote the project in MATLAB, so I am unsure how many changes would be needed to make the project run in Octave.

# Using Neural Networks: Digit Recognition

To use this software, you should:

1. Download all of the files in this repo to the same directory on your computer.
2. Run one of the following programs in Octave/MATLAB:
	- "ex3.m" - located in the "Forward Propagation" folder, this program classifies handwritten digits using only logistic regression.
	- "ex3_nn.m" - located in the "Forward Propagation" folder, this program classifies handwritten digits using a neural network with pre-defined, well-performing parameters.
	- "ex4.m" - located in the "Backward Propagation" folder, this program classifies handwritten digits using a neural network that learns appropriate parameters.

The provided data sets have 10 classes (one for each digit 0-9).  However, the code is written to easily run with any number of classes, and provides vectorized implementation where possible (i.e., avoids for-loops).

NOTE: This program is untested on new examples.  I have trained the neural network on the (included) training data sets and it performs quite well, but have not yet constructed or found any new data sets with which to test the network.

Feel free to make changes to the code or load your own data sets into the programs.

*The following sections describe the files and program flow of each program more in-depth.*

## Part 1: Comparison of Logistic Regression and Neural Networks (Forward Propagation)

Part 1 of this project compares logistic regression to a neural network for the task of recognizing handwritten digits (the arabic numerals 0 to 9).  The activation function for the neural network is sigmoid.

Part 1 is found in the "Forward Propagation" Folder of this repo.

### Files in the Forward Propagation Folder

1. ex3.m - script that prepares data and calls functions to classify digits using only logistic regresison (and not neural networks)
2. ex3_nn.m - script that prepares data and calls functions to classify digits using a neural network with pre-determined, well-performing parameters
3. ex3data1.mat - Training data set of hand-written digits
4. ex3weights.mat - Initial weights for the neural network 
5. displayData.m - Function to help visualize the dataset
6. fmincg.m - Function minimization routine (similar to fminunc)
7. sigmoid.m - Sigmoid function
8. [⋆] lrCostFunction.m - Logistic regression cost function
9. [⋆] oneVsAll.m - Script that trains a one-vs-all multi-class classifier
10. [⋆] predictOneVsAll.m - Logistic regression one-vs-all multi-class classifier
11. [⋆] predict.m - Neural network one-vs-all multi-class classifier

[*] indicates a file a in which I wrote substantial code.  The other files were provided by the professor (Andrew Ng) and I have only modified them slightly if at all.

### Forward Propagation Program Flow

To classify handwritten digits using logsitic regression, you should open and run the file named "ex3.m" in MATLAB/Octave.  ex3.m does the following:

1. Loads the data and visualizes data
	- ex3.m loads data into memory and visualizes some of the training set.
		- Loads 5000 training examples of handwritten digits, saved in a native Octave/MATLAB matrix format.
			- Each training example is a 20 pixel by 20 pixel grayscale image of a digit from 0-9.
			- Each pixel is represented by a floating point number indicating the grayscale intensity at that location.
			- The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector, which gives a 5000 by 400 matrix where every row is a training example for a handwritten digit image.
		- Loads a 5000-dimensional vector that contains labels for the training set.
		- Randomly selects 100 training examples and displays the images together.
3. Trains logistic regression classifiers
	- ex3.m trains one logistic regression classifier for each numerical digit
		- Cost function is regularized (except for the bias term).
		- Returns all the classifier parameters in a matrix, where each row corresponds to the learned logistic regression parameters for one class.
		- Uses the fmincg optimization function.
4. Classifies the training set using the trained logistic regression classifiers
	- ex3.m predicts the digit contained in a given image using the trained logistic regression classifiers. 
		- Predicts the digit corresponding to the logistic regression classifier that outputs the highest probability.
		- Predicts the training set examples with 95.02% accuracy.

To classify handwritten digits using a neural network that has already been trained, you should open and run the file named "ex3_nn.m" in MATLAB/Octave.  ex3_nn.m does the following:

1. Loads and visualizes data
	- ex3_nn.m loads and visualizes data in much the same way as ex3.m.
2. Loads neural network parameters
	- ex3_nn.m loads parameters for the neural network that have already been trained on this data set.
3. Classifies the training set using a neural network
	- ex3_nn.m implements feedforward propagation in a neural network using the pre-trained parameters to classify the digit contained in a given image.
		- The nerual network has three layers: an input layer, a "hidden" layer, and an output layer.
		- Since each image has 400 pixels, the input layer has 400 units.
		- The hidden layer has 25 units.
		- The output layer has 10 units (corresponding to the 10 digit classes).
		- Predicts the training set examples with 97.52% accuracy (2.5% improvement over the pure logistic regression classifier).
4. Visualizes the predictions
	- ex3_nn.m displays images from the training set one at a time while the console prints out the predicted label for the displayed image.

## Part 2: Backward Propagation

Part 2 of this project classifies handwritten digits using a neural network that *learns* parameters for the given data set (so, it implements the backpropagation algorithm).  The activation function for the neural network is sigmoid.

Part 2 is found in the "Backward Propagation" Folder of this repo.

### Files in the Backward Propagation Folder

1. ex4.m - Script that loads data and calls functions
2. ex4data1.mat - Training data set of hand-written digits
3. ex4weights.mat - Initial parameters for the neural network 
4. displayData.m - Function to help visualize the dataset
5. fmincg.m - Function minimization routine (similar to fminunc)
6. sigmoid.m - Sigmoid function
7. computeNumericalGradient.m - Function to numerically compute gradients
8. checkNNGradients.m - Function to help check gradients
9. debugInitializeWeights.m - Function for initializing weights 
10. predict.m - Neural network prediction function
11. [⋆] sigmoidGradient.m - Computes the gradient of the sigmoid function
12. [⋆] randInitializeWeights.m - Randomly initializes weights (parameters)
13. [⋆] nnCostFunction.m - Neural network cost function

[*] indicates a file in which I wrote substantial code.  The other files were provided by the professor (Andrew Ng) and I have only modified them slightly if at all.

### Backward Propagation Program Flow

To classify handwritten digits using a neural network that learns parameters from a given data set, you should open and run the file named "ex4.m" in MATLAB/Octave.  ex4.m works with data sets of any size and for any number of classes K such that K >= 3.

ex4.m does the following:

1. Loads and visualizes the data
	- ex4.m loads the training data set and displays it on a 2-dimensional plot.
		- Same dataset used for Part 1 (Forward Propagation).  See above for a more in-depth explanation of the data set.
2. Loads initial debugging parameters
	- ex4.m loads initial parameters that help check if the neural network functions properly.
		- The parameters fit a neural network with the same structure as in Part 1 (Forward Propagation):
			- Three layers: an input layer, a "hidden" layer, and an output layer.
			- Since each image has 400 pixels, the input layer has 400 units.
			- The hidden layer has 25 units.
			- The output layer has 10 units (corresponding to the 10 digit classes).
		- The parameters are the same as given in Part 1 (Forward Propagation), and so are pre-trained on the given data set.
3. Computes unregularized cost
	- ex4.m computes and checks the unregularized cost of feeding the training set forward through the neural network using the given debugging parameters.
		- Demonstrates an intermediate step on the way to implementing a regularized cost function that helps learn paramaters during backpropagation.
4. Computes regularized cost
	- ex4.m computes and checks the regularized cost of feeding the training set forward through the neural network using given debugging parameters.
		- Does not regularize the bias terms.
		- Demonstrates the regularized cost function that helps learn paramaters during backpropagation.
5. Computes a test sigmoid gradient
	- ex4.m computes and checks the gradient of the sigmoid function evaluated for test input.  
		- Demonstrates an intermeditate step that helps calculate the amount of the total error that should be attributed to each unit in the neural network.
		- Intuitively, this value represents the impact that changing a unit's input would make on the amount of the total error attributed to that unit.
6. Randomly initializes parameters
	- ex4.m randomly initializes starting parameters to be fed into the neural network for trainig.
7. Tests the backpropagation algorithm on a small neural network
	- ex4.m computes gradients on a small test neural network to check if backpropagation is working correctly.
		- Runs a “forward pass” to compute all the activations throughout the network, including the output values.
		- Computes an “error term” δ for each output and hidden unit which measures how much that unit was “responsible” for any errors in the output.
			- For an output unit, δ is a function of the unit's input and the difference between the unit's activation and the target value ("did we get the right classification?").
			- For a hidden unit, δ is a function of the unit's input and the error terms of the units in the next layer to which it directly connects weighted by the strengths of those connections (i.e., the parameters). 
		- Accumulates the parameter gradient for each training example for each unit of the test.
		- Calculates the (unregularized) gradients by dividing the accumulated gradients by the number of training examples.
8. Tests the backpropagation algorithm with numerical gradient checking
	- ex4.m computes gradients on a small nueral network and verifies gradients using numerical gradient checking.
		- Uses a small value ε to numerically compute the slope of a line formed by the points (parameter-ε,cost(parameter-ε)) and (parameter+ε,cost(parameter+ε))
		- Assuming ε = 10^(−4) and a functioning backpropagation algorithm, the backpropagation gradient and numerical gradient will agree to at least four significant digits.
9. Regularizes the neural network
	- ex4.m regularizes the gradient of the neural network.
		- Does not regularize bias parameters.
10. Learns parameters for the neural netowrk
	- Using the regularized cost function and gradient, ex4.m learns parameters for the neural network using the optimization function fmincg.
11. Visualizes the hidden layer
	- ex4.m visualizes the hidden layer in the trained network by displaying the highest activation value for each unit in the hidden layer.
		- In this context, "highest activation value for a unit" means "the combination of pixel intensities in a 20x20 greyscale image that gives a simoid function value closest to "1" for that unit".
12. Classifies the training set
	- ex4.m will classify each example in the training set according to the parameters learned from the neural network and report its accuracy and cost using 1) the given debugging parameters, 2) the randomly initialized parameters, and finally 3) the learned parameters.
		- Using the randomly initialized parameters and 0 steps of the optimization function, the neural network predicts the digit from the handwritten input with about 10.0% accuracy (which may vary due to the randomization).
		- Using the learned parameters and 400 steps of the optimization function, the neural network predicts the digit from the handwritten input with about 99.5% accuracy (which may vary slightly due to the random initialization).
		- User can adjust accuracy of training set classification by training the neural network for a different number of iterations or with a different regularization parameter λ. This can be helpful to avoid "overfitting" a training set if the neural network does not perform well on new examples that it has not seen before.

# License

This software is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl.html).  Accordingly, you are free to run, study, share, and modify this software only if you give these same freedoms to users of *your* implementation of this software.

# Credit

This README is based on 1) information from the problem definition PDFs provided by Andrew Ng at Stanford University for the class [Machine Learning](https://www.coursera.org/learn/machine-learning) offered on Coursera and 2) my experience writing the software specified by the PDFs.
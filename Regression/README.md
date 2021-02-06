# Regression Codes (UNDER CONSTRUCTION)

## Introduction

### Linear Regression

Linear regression is one of the simplest machine learning algorithms, and because of this, it is often the one that is first taught in a machine learning course or textbook. Besides its simplicity, there are several other useful reasons to start a machine learning course with linear regression.  First, though it is quite simple it has all of the elements of a machine learning algorithm, so parallels can be drawn to it when learning more complicated algorithms.  Second, it is rather straight forward to derive an analytical expression for the "learned" parameters of the algorithms (which will be done below).  Third, most people are familar with linear regression since it is often also taught in statistics courses.  And, finally, since linear regression is a simple algorithm, it is straightforward to write a code that implements it.

The output of a linear regression algorith can be writtens as follows:
![equation](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20%3D%20X%5Ctheta).

The variable ![equation](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D), called "y-hat", used used to represent the output of the linear regression algorithm (and most machine learning algorithms).  In this context ![equation](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D) is a vector.  X is a matrix known as the design matrix.  The design matrix is used to feed the input data into the linear regression algorithm.  The design matrix can be as simple as just a matrix containing all of the input for the algorithm.  For example, to fit a linear regression algorithm to a data set with two input variables, f(x, y), the design matrix could look as follows:

![equation](https://latex.codecogs.com/gif.latex?X%20%3D%20%5Cbegin%7Bbmatrix%7D%20x_0%20%26%20y_0%20%5C%5C%20x_1%20%26%20y_1%20%5C%5C%20x_2%20%26%20y_2%20%5C%5C%20.%20%26%20.%20%5C%5C%20.%20%26%20.%20%5C%5C%20.%20%26%20.%20%5C%5C%20x_%7Bn-1%7D%20%26%20y_%7Bn-1%7D%5C%5C%20%5Cend%7Bbmatrix%7D)

The design matrix can also be used to encode an prior knowledge data into the linear regression algorithm.  For example, if you know the data set f(x,y) should be roughly f(x,y) ![equation](https://latex.codecogs.com/gif.latex?%5Capprox) then more helpful design matrix than the one above may be:

![equation](https://latex.codecogs.com/gif.latex?X%20%3D%20%5Cbegin%7Bbmatrix%7D%20x_0y_0%20%5C%5C%20x_1y_1%20%5C%5C%20x_2y_2%20.%20%5C%5C%20.%20%5C%5C%20.%20%5C%5C%20x_%7Bn-1%7Dy_%7Bn-1%7D%20%5Cend%7Bbmatrix%7D)

The final variable in the linear regression, ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta), is a set of weights that are "training" by the linear regression algorithm to make X![equation](https://latex.codecogs.com/gif.latex?%5Ctheta) as close to the true data set as possible.

The error of the linear regression algorithm is measured by a loss function.  For linear regression, the loss function is the standard mean-squared error function.  The loss function for the linear regression algorithm can be written as a sum or in matrix-vector form as follows:

![equation](https://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D0%7D%5E%7Bn-1%7D%20%28y_i%20-%20%5Chat%7By%7D_i%29%5E2%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5B%28y-X%5Ctheta%29%28y-X%5Ctheta%29%5D)

Finding the values of ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta) that will best work in the algorithm is as simple as finding the values of ![equation](![equation](https://latex.codecogs.com/gif.latex?%5Ctheta) that will minimize the loss function.  By taking the partial derivative of the loss function with respect to ![equation](![equation](https://latex.codecogs.com/gif.latex?%5Ctheta) and setting it equal to zero, we get:

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%7D%20%3D%200%20%3D%20X%5ET%28y-X%5Ctheta%29)

Finally, the optimal values of ![equation](![equation](https://latex.codecogs.com/gif.latex?%5Ctheta) are found by solving the above equation for ![equatip(![equation](https://latex.codecogs.com/gif.latex?%5Ctheta).

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta%20%3D%20%28X%5ETX%29%5E%7B-1%7DX%5ETy)


### Ridge Regression

### Lasso Regression

### Kernel Ridge Regression


## Codes 

### [Linear Regression](LinearRegression.py)

### [Ridge Regression](RidgeRegression.py)

### [Kernel Ridge Regression](KernelRidgeRegression.py)

### [Regression Support](RegressionSupport.py)

### [Regression Test Code](RegressionTestCode.py)

This is an all inclusive code to test extrapolating with Linear Regression, Ridge Regression, and Kernel Ridge Regression using both regular format training and sequential format training.  This code also includes options for graphing and saving the results.  For a given data set the code will perform hyperparameter tuning on whatever combination of linear regression, ridge regression, kernel ridge regression, regular data formatting, and sequential data formatting that are specified by setting the values of booleans.  The code will print the optimized parameters once found and the best MSE for the extrapolation to the terminal if the verbose boolean is set to True.  Graphing capabilities and saving parameters and extrapolated data sets are also controlled with booleans.

### [Convergence Regression](ConvergenceRegression.py)

## Future Improvements

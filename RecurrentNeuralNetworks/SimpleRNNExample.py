##################################################
# Simple RNN Example
# Julie Butler Hartley
# Date Created: October 22, 2020
# Last Modified: October 25, 2020
# Version 1.0.0
#
# A code that utilizes a simple recurrent neural network from Keras and time series
# forecasting data formatting to perform extrapolations.
##################################################

##############################
# IMPORTS
##############################
# SYSTEM LEVEL IMPORTS
import sys

# THIRD-PARTY IMPORTS
# For matrices and calculations
import numpy as np
# For machine learning (backend for keras)
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior
# User-friendly machine learning library
# Front end for TensorFlow
import keras 
# Different methods from Keras needed to create an RNN
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation 
from keras.layers.recurrent import SimpleRNN
# For graphing
import matplotlib.pyplot as plt

# Changing the import directory
sys.path.append('../DataSets/')
# LOCAL IMPORTS
# Encoded data sets but can apply this code to any data set
from Datesets import *


##############################
# DATA SET
##############################
# Get a string that represents the name of the data set, a recommended training 
# dimension for the data, the total x data, and the total y data for a data set
# the is encoded in the file DataSets.py
# This code can be used with other data sets as long as a training dimension is supplied
# with the name "dim", the x data is in a one dimensional numpy array named "X_tot", and the
# y data is in a one dimensional numpy array called "y_tot".
name, dim, X_tot, y_tot = VaryDimension()
# Check to see if the data set is complete
assert len(X_tot) == len(y_tot)


##############################
# FORMAT_DATA
##############################
# This is the method that takes the y data from the data set and formats it using sequential or time
# series forcasting.  The method can format data using any length of sequence, but the rest of the 
# code requires a sequence length of 2.  
# FORMAT_DATA
def format_data(data, length_of_sequence = 2):  
    """
        Inputs:
            data(a numpy array): the data that will be the inputs to the recurrent neural
                network
            length_of_sequence (an int): the number of elements in one iteration of the
                sequence patter.  For a function approximator use length_of_sequence = 2.
        Returns:
            rnn_input (a 3D numpy array): the input data for the recurrent neural network.  Its
                dimensions are length of data - length of sequence, length of sequence, 
                dimnsion of data
            rnn_output (a numpy array): the training data for the neural network
        Formats data to be used in a recurrent neural network.  The resulting data points have the
        following format for a sequence length of n: ((y1, y2, ..., yn), yn+1).  This function is 
        adapted from the one found here: 
        https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    """
    # To store the formated "x" and "y" data
    X, Y = [], []
    for i in range(len(data)-length_of_sequence):
        # Get the next length_of_sequence elements
        a = data[i:i+length_of_sequence]
        # Get the element that immediately follows that
        b = data[i+length_of_sequence]
        # Reshape so that each data point is contained in its own array
        a = np.reshape (a, (len(a), 1))
        X.append(a)
        Y.append(b)
    # Convert into numpy arrays as these are easier to use later in the code.
    rnn_input = np.array(X)
    rnn_output = np.array(Y)
    return rnn_input, rnn_output

##############################
# TRAINING DATA
##############################
# Generate the training data for the RNN using a sequence length of 2
# Get the first dim points from the total data set to use for training
X_train = X_tot[:dim]
y_train = y_tot[:dim]
# Formating the y component of the training data using the time series forecasting 
rnn_input, rnn_training = format_data(y_train, 2)

##############################
# RNN
##############################
def rnn(length_of_sequences, batch_size = None, stateful = False):
    """
        Inputs:
            length_of_sequence (an int): the length of sequence used to format the training 
                data (i.e. the length of the sequence used in format_data).
            batch_size (an int): See Keras documentation for Input
            stateful (a boolean): See Keras documentation for SimpleRNN
        Returns:
            model (a Keras model): the build and compiled recurrent neural network
        Creates a simple recurrent neural network with one simple recurrent hidden layer with
        200 hidden neurons and compiles the network using a mean-squared error loss function 
        and an Adam's optimizer.
    """
    # Number of neurons in the input and output layer
    in_out_neurons = 1
    # Number of neurons in the hidden layer
    hidden_neurons = 200
    # Create the input layer
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons))  
    # Create the simple recurrent hidden layer
    rnn = SimpleRNN(hidden_neurons, 
                    return_sequences=False,
                    stateful = stateful,
                    name="RNN")(inp)
    # Create a dense (feedforward) neural network layer which will act as the output layer
    dens = Dense(in_out_neurons,name="dense")(rnn)
    # Build the model
    model = Model(inputs=[inp],outputs=[dens])
    # Compile the model using the specified loss function and optimizer
    model.compile(loss="mean_squared_error", optimizer="adam")  
    return model

##############################
# CREATE THE RECURRENT NEURAL NETWORK
##############################
## use the default values for batch_size, stateful
model = rnn(length_of_sequences = rnn_input.shape[1])
# Print a summary of the Keras model to the console
model.summary()

##############################
# TRAIN THE RECURRENT NEURAL NETWORK
##############################
# Fit the model using the training data created above using 150 training iterations and a
# validation split of 0.05
hist = model.fit(rnn_input, rnn_training, batch_size=None, epochs=150, 
                 verbose=True,validation_split=0.05)


##############################
# LOSS FUNCTION GRAPH
##############################
# Creates a graph of the training loss/error and the validation loss/error as a function of the
# number of training iterations performed.  This is useful to make sure the model is not 
# overtraining.
# Get the data from the trained model and plot it
for label in ["loss","val_loss"]:
    plt.plot(hist.history[label],label=label)
# Label the x axis, the y axis, and add a title
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("The final validation loss: {}".format(hist.history["val_loss"][-1]))
# Add a legend then show the plot
plt.legend()
plt.show()

##############################
# TEST_RNN
##############################
def test_rnn (x_known, y_known):
    """
        Inputs: 
            x_known (a list or numpy array): the known x data, likely X_tot imported using
                the data set.
            y_known (a list or numpy array): the known y data, likely y_tot imported using
                the data set.
        Returns:
            None.
        Extrapolates from the training data to a complete data set using the trained recurrent
        neural network.  Performs data analysis on the predicted data points and creates a graph
        of the known data and the predicted data.
    """
    # Segment off the training data from the known data
    y_pred = y_known[:dim].tolist()
    # Create the first point that will be used to predict the following point using the trained
    # recurrent neural network.  In this case the first point contains the two points if the 
    # training data, so the first point that will be predicted is the first point to fall sequentially
    # after the training data
    next_input = np.array([[[y_test[dim-2]], [y_test[dim-1]]]])
    # Save the last number in the prediction point for later use
    last = [y_test[dim-1]]
    # Loop until the predicted data set is the same length as the known data set.
    for i in range (dim, len(y_known)):
        # Predict the next point and add it to the predicted data set
        next = model.predict(np.asarray(next_input))
        y_pred.append(next[0][0])
        # print the difference between the predicted point and the correspinding known point
        print ('DIFF: ', next[0][0]-y_known[i])
        # Create the point that will be uses to make a prediction on the next interation
        next_input = np.array([[last, next[0]]], dtype=np.float64)
        last = next
    # Print the MSE of the predicted data and the known data.  This is a measure of how well the 
    # extrapolation worked.
    print('MSE: ', np.square(np.subtract(y_known, y_pred)).mean())
    # Save the predicted data set as a csv file for future use
    name = datatype + 'Predicted'+str(dim)+'.csv'
    np.savetxt(name, y_pred, delimiter=',')
    # Plot both the known and the predicted data sets and add a legend
    fig, ax = plt.subplots()
    ax.plot(x_known, y_known, label="true", linewidth=3)
    ax.plot(x_known, y_pred, 'g-.',label="predicted", linewidth=4)
    ax.legend()
    # Create a semi-transparent red box to represent the training data
    ax.axvspan(x_known[0], x_known[dim], alpha=0.25, color='red')
    plt.show()
 
##############################
# PREDICT NEW POINTS
##############################
# Predict the remaining points to finish the data set
test_rnn(X_tot, y_tot)

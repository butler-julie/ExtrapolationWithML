##################################################
# Simple RNN Example
# Julie Butler Hartley
# Date Created: October 22, 2020
# Last Modified: October 22, 2020
# Version 0.8.0
#
# A code that utilizes a simple recurrent neural network from Keras and time series
# forecasting data formatting to perform extrapolations.
##################################################

##############################
# IMPORTS
##############################
# THIRD_PARTY IMPORTS
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
# For timing purposes
from timeit import default_timer as timer
# For graphing
import matplotlib.pyplot as plt

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
        Formats data to be used in a recurrent neural network.
    """

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
    in_out_neurons = 1
    hidden_neurons = 200
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons))  
    rnn = SimpleRNN(hidden_neurons, 
                    return_sequences=False,
                    stateful = stateful,
                    name="RNN")(inp)
    dens = Dense(in_out_neurons,name="dense")(rnn)
    model = Model(inputs=[inp],outputs=[dens])
    model.compile(loss="mean_squared_error", optimizer="adam")  
    return model



##############################
# CREATE THE RECURRENT NEURAL NETWORK
##############################
## use the default values for batch_size, stateful
model = rnn(length_of_sequences = rnn_input.shape[1])
#model.summary()

start = timer()
hist = model.fit(rnn_input, rnn_training, batch_size=None, epochs=150, 
                 verbose=True,validation_split=0.05)



#for label in ["loss","val_loss"]:
#    plt.plot(hist.history[label],label=label)

#plt.ylabel("loss")
#plt.xlabel("epoch")
#plt.title("The final validation loss: {}".format(hist.history["val_loss"][-1]))
#plt.legend()
#plt.show()

def test_rnn (x1, y_test, plot_min, plot_max):
    #y_pred = [y_test[0], y_test[1]]
    #current_pred = np.array([[[y_test[0]], [y_test[1]]]])
    #last = np.array([[y_test[1]]])
    #for i in range (2, len(y_test)):
    #    next = model.predict(current_pred)
    #    y_pred.append(next)
    #    current_pred = np.asarray([[last.flatten(), next.flatten()]])
    #    last = next

    #assert len(y_pred) == len(y_test)
        
    #plt.figure(figsize=(19,3))
    #plt.plot([10, 10, 10], [1.5, 0, -1.5])

    #X_test, a = format_data (y_test.copy(), 2)
    #print X_test[0]
    #y_pred = model.predict(X_test)

    #x1 = x1[2:]
    #y_test = y_test[2:]

    y_pred = y_test[:dim].tolist()
    next_input = np.array([[[y_test[dim-2]], [y_test[dim-1]]]])
    print(next_input)
    last = [y_test[dim-1]]

    for i in range (dim, len(y_test)):
        print ('ITER: ', i)
        #print (type(next_input[0][0][0]))
        next = model.predict(np.asarray(next_input))
        y_pred.append(next[0][0])
        #print(next)
        print ('DIFF: ', next[0][0]-y_test[i])
        next_input = np.array([[last, next[0]]], dtype=np.float64)
        last = next
        print(type(last))

    print('MSE: ', np.square(np.subtract(y_test, y_pred)).mean())
    name = datatype + 'Predicted'+str(dim)+'.csv'
    np.savetxt(name, y_pred, delimiter=',')
    fig, ax = plt.subplots()
    ax.plot(x1, y_test, label="true", linewidth=3)
    ax.plot(x1, y_pred, 'g-.',label="predicted", linewidth=4)
    ax.legend()

    ax.axvspan(plot_min, plot_max, alpha=0.25, color='red')
    plt.show()
    
    #diff = y_test - y_pred.flatten()

    #plt.plot(x1, diff, linewidth=4)
    #plt.show()

addition = 3

test_rnn(X_tot, y_tot, X_tot[0], X_tot[dim-1])
end = timer()
print('Time: ', end-start)


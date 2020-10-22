#################################################
# Regression Support
# Julie Butler Hartley
# Version 0.1.0
# Date Created: July 7, 2020
# Last Modified: August 17, 2020
#################################################

#################################################
# OUTLINE
# Helper Functions
# time_series_data(data, length_of_sequence = 2)
# mse (A, B)
#
# Graphing Functions
# graph_LR_RR_KRR (X_tot, y_tot, isKR, kr_data, isR, r_data, isLR, lr_data,
#   isSave, savename, isDisplay)
# graph_reg_vs_seq (X_tot, y_tot, isRegular, reg_data, isSequential, seq_data,
#   isSave, savename, isDisplay)
#################################################

#############################
# IMPORTS
#############################
# THIRD PARTY IMPORTS
# For array handling
import numpy as np
# For graphing
import matplotlib.pyplot as plt


#################################################
#
# HELPER FUNCTIONS
#
#################################################

#############################
# TIME_SERIES_DATA
#############################
def time_series_data(data, length_of_sequence = 2):
    """
        Inputs:
            data(a numpy array): the data that will be the inputs to the recurrent neural
                network
            length_of_sequence (an int): the number of elements in one iteration of the
                sequence patter.  For a function approximator use length_of_sequence = 2.
        Returns:
            rnn_input (a 2D numpy array): the input data for the regression.
            rnn_output (a numpy array): the training data for the regression
        Formats data in a time series-inspired way to be used in ridge and kernel ridge
        regression.
        NOTE: formally format_data_dnn from RNN codes.
    """
    # Lists to hold the formatted data
    inputs, outputs = [], []
    # Loop through the data
    for i in range(len(data)-length_of_sequence):
        # Get the next length_of_sequence elements
        a = data[i:i+length_of_sequence]
        # Get the element that immediately follows that
        b = data[i+length_of_sequence]
        # Add new points to the returned arrays
        inputs.append(a)
        outputs.append(b)
    # Format the lists as numpy arrays
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    # Return the two formatted arrays
    return inputs, outputs

#############################
# MSE (Mean-Squared Error)
############################
def mse (A, B):
    """
        Inputs:
            A, B (numpy arrays of the same length): two different data sets
        Returns:
            Unnamed (a float): the mean-squared error score between data sets A and B
        Finds the mean-squared error of the two data sets given as inputs.
    """
    return ((A-B)**2).mean()


#################################################
#
#  GRAPHING FUNCTIONS
#
#################################################

#############################
# GRAPH KRR AND RR
#############################
def graph_LR_RR_KRR (X_tot, y_tot, isKR, kr_data, isR, r_data, isLR, lr_data,
    isSave, savename, isDisplay):
    """
        Inputs:
            X_tot (a list or numpy array): the x component of the data set
            y_tot (a list or numpy array): the y component of the data set
            isKR (a boolean): True if kernel ridge regression data is being
                supplied, False otherwise.
            kr_data (a list or numpy array): the data that was extrapolated using
                kernel ridge regression.  If isKR is False then pass an empty list
            isR (a boolean): True if ridge regression daa is being supplied, False
                otherwise.
            r_data (a list or numpy array): the data that was extrapolated using
                ridge regression.  If isR is False then pass an empty list.
            isLR (a boolean): True id linear regression data is being supplied,
                False otherwise
            lr_data (a list or numpy array): the data that was extrapolated using
                linear regression.  If isLR is False then pass an empty list.
            isSave (a boolean): True if the graph is to be saved.
            savename (a string): the name to save the graph in.  Pass an empty
                string if isSave is False.
            isDisplay (a boolean): True if the graph is to be displayed.
        Returns:
            None.
        Plots extrapolated data from kernel ridge regression, ridge regression,
        and linear regression on the same plot.  Can supply all three data sets,
        or only a subset.  Options avaliable to save and display the graph once
        created.
    """
    # If data is given from a kernel ridge regression algorithm
    if isKR:
        plt.plot(X_tot, kr_data, linewidth=4, label='Kernel Ridge Regression')
    # If data is given from a ridge regression algorithm
    if isR:
        plt.plot(X_tot, r_data, linewidth=4, label='Ridge Regression')
    # If data is given from a linear regression algorithm
    if isLR:
        plt.plot(X_tot, lr_data, linewidth=4, label='Linear Regression')
    # Plot the actual data, using a fixed line style with larger size to be
    # more noticeable
    plt.plot(X_tot, y_tot, '^', linewidth=2, label='True Data')
    plt.legend(loc='best')
   # Save the graph if wanted
    if isSave:
        plt.savefig(savename)
    # Display the graph if wanted
    if isDisplay:
        plt.show()

def graph_reg_vs_seq (X_tot, y_tot, isRegular, reg_data, isSequential, seq_data,
    isSave, savename, isDisplay):
    """
        Inputs:
            X_tot (a list or numpy array): the x component of the data set
            y_tot (a list or numpy array): the y component of the data set
            isRegular (a boolean): True case means that data from the regular
                training algorithm is being supplied.
            reg_data (a list or Numpy array): Data extrapolated from a model
                trained using the regular algorithm.  If isRegular is False pass
                None.
            isSequential (a boolean): True case means that data from the sequential
                training method is being supplied.
            seq_data (a list or Numpy array): Data extrapolated from a model
                trained using the sequential algorithm.  If isRegular is False pass
                None.
            isSave (a boolean): True if the graph is to be saved.
            savename (a string): the name to save the graph in.  Pass an empty
                string if isSave is False.
            isDisplay (a boolean): True if the graph is to be displayed.
        Returns:
            None.

        Plots extrapolated data using the regular training method and/or the
        sequential training method.  Can supply both data sets or only one.
        Options avaliable to save and display the graph once
        created.
    """
    # If the results from a regular training algorithm are supplied
    if isRegular:
        plt.plot (X_tot, reg_data, linewidth=4, label='Regular Training Format')
    # If the results from a sequentual training algorithm are supplied
    if isSequential:
        plt.plot (X_tot, seq_data, linewidth=4, label='Sequential Training Format')
    # Plot the complete (given) data set for comparison
    plt.plot(X_tot, y_tot, '^', linewidth=6, label='True Data')
    # Add the legend at the best location
    plt.legend(loc='best')
   # Save the graph if wanted
    if isSave:
        plt.savefig(savename)
    # Display the graph if wanted
    if isDisplay:
        plt.show()

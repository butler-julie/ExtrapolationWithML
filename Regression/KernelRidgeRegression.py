#################################################
# Kernel Ridge Regression
# Julie Butler Hartley
# Version 1.0.0
# Date Created: July 7, 2020
# Last Modified: July 14, 2020
#
# A collection of methods for performing kernel ridge regression of data sets, both with
# and without time series formatting.
#
# TO DO
# finish documentation 
# fix warnings
#################################################

#################################################
# OUTLINE:
#
#################################################


#############################
# IMPORTS
#############################
# THIRD-PARTY IMPORTS
# For array management and some calculations
import numpy as np
# Used for timing the running of codes
import time
# Kernel Ridge Methods
from sklearn.kernel_ridge import KernelRidge
# For plotting
import matplotlib.pyplot as plt
# Prevents extraneous printing of messages during a grid search
import warnings
# For making parameter lists in hyperparameter tuning
from itertools import product

# LOCAL IMPORTS
from RegressionSupport import *

#################################################
# KERNEL RIDGE REGRESSION
#################################################
class KernelRidgeRegressionAnalysis ():

    ##############################
    # INIT
    ##############################
    def __init__ (self):
        print ("Kerenl Ridge Regression instance started.")

    ##############################
    # STR
    ##############################
    def __str__ (self):
        return "Instance of Kernel Ridge Regression class"
    ##################################################
    #
    # COMPLETE DATA SET IS KNOWN
    #
    ##################################################

    #############################
    # KNOWN DATA 
    #############################
    def known_data (self, X_train, y_train, y_tot, training_dim, params, verbose=True):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_tot (a list of numpy array): the total set of data points (training plus validation)
                training_dim (an int): the size of the traing data (i.e. the number of points
                    from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.  Default value is true
            Returns:
                y_return (a list): the known points and the extrapolated data points
                Unnamed (a float): the MSE error between the true data and the predicted
                    data
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set.  Prints the MSE 
            score of the extrapolated data set compared to the true data set if desired and
            then returns the extrapolated data set.
                    
        """
        # To ensure that all parameters are present 
        assert len(params)==5

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], 
            coef0=params[3], gamma=params[4])    

        # Fit the model to the training data
        kr.fit(X_train, y_train)

        # Use the trained model to predict the points in the validation set
        y_return = y_tot[:training_dim].tolist()
        next_input = [[y_return[-1]]]
        last = y_return[-1]
        while len(y_return) < len(y_tot):
            try:
                next = kr.predict(next_input)
            except:
                print ('Overflow encountered on predicton')
                return None, 1e10
            y_return.append(next[0])
            next_input =[[next[0]]]

        # Print the MSE error if needed
        if verbose:  
            print ('KERNEL RIDGE MSE VALUE: ', mse(y_tot, y_return))

        # Return the predicted points and the MSE error
        return y_return, mse(y_tot, y_return)

    #############################
    # KNOWN DATA CR (CONTINUOUS RETRAIN)
    #############################
    def known_data_cr (self, X_train, y_train, y_tot,
        training_dim, params, verbose):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_tot (a list of numpy array): the total set of data points
                training_dim (an int): the size of the traing data (i.e. the number of 
                    points from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.   
            Returns:
                y_return (a list): the known points and the extrapolated data points
                Unnamed (a float): the MSE error between the true data and the predicted
                    data
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set. Kernel ridge 
            regression is performed after each point is extrapolated to hopefully decrease 
            the average MSE score.  Prints the MSE score of the extrapolated data set 
            compared to the true data set if desired and then returns the extrapolated data
            set.
                    
        """
        # To ensure that all parameters are present
        assert len(params)==5

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], coef0=params[3], gamma=params[4])

        # Add the known training data to the predicted points list
        y_return = y_tot[:training_dim].tolist()
        
        # While the length of the predicted points list is less than the total number of 
        # needed points
        while len(y_return) < len(y_tot):
            # Ensure that there are enough points the the predicted points list to be 
            # properly formatted.  Re-fitting the model only occurs when there are enough 
            # data points for the data to be properly formatted
            if len(y_return) % seq == 0:
                # Format the data
                X_train, y_train = time_series_data(y_return)
                # Fit the model
                kr.fit(X_train, y_train)
            # Predict the next point in the data set and add it to the list
            next_input = [[ y_return[-1]]]    
            next = kr.predict(next_input)    
            y_return.append(next[0])

        # Print the MSE error if needed
        if verbose:
            print ('KERNEL RIDGE CONTINUOUS RETRAIN MSE VALUE: ', mse(y_tot, y_return))

        # Return the predicted list
        return y_return, mse(y_tot, y_return)

    #############################
    # KNOWN DATA SEQ (SEQUENTIAL)
    #############################
    def known_data_seq (self, X_train, y_train, y_tot, training_dim, params, verbose=True, seq=2):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_tot (a list of numpy array): the total set of data points (training plus validation)
                training_dim (an int): the size of the traing data (i.e. the number of points
                    from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.
            Returns:
                y_return (a list): the known points and the extrapolated data points
                Unnamed (a float): the MSE error between the true data and the predicted
                    data
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set.  Prints the MSE 
            score of the extrapolated data set compared to the true data set if desired and
            then returns the extrapolated data set.
                    
        """
        # To ensure that all parameters are present 
        assert len(params)==5

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], coef0=params[3], gamma=params[4])    

        # Fit the model to the training data
        kr.fit(X_train, y_train)

        # Use the trained model to predict the points in the validation set
        y_return = y_tot[:training_dim].tolist()
        next_input = [[y_return[-2], y_return[-1]]]
        last = y_return[-1]
        while len(y_return) < len(y_tot):
            try:
                next = kr.predict(next_input)
            except:
                print ('Overflow encountered on predicton')
                return None, 1e10
            y_return.append(next[0])
            next_input =[[last, next[0]]]
            last = next[0]

        # Print the MSE error if needed
        if verbose:  
            print ('KERNEL RIDGE MSE VALUE: ', mse(y_tot, y_return))

        # Return the predicted points and the MSE error
        return y_return, mse(y_tot, y_return)

    #############################
    # KNOWN DATA CR SEQ (CONTINUOUS RETRAIN, SEQUENTIAL)
    #############################
    def known_data_cr_ts (self, X_train, y_train, y_tot,
        training_dim, params, verbose, seq=2):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_tot (a list of numpy array): the total set of data points
                training_dim (an int): the size of the traing data (i.e. the number of points
                    from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.
                seq (an int): the length of the series to use in the time series formatting (default 
                    value is 2)    
            Returns:
                y_return (a list): the known points and the extrapolated data points
                Unnamed (a float): the MSE error between the true data and the predicted
                    data
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set. Kernel ridge 
            regression is performed after each point is extrapolated to hopefully decrease 
            the average MSE score.  Prints the MSE score of the extrapolated data set 
            compared to the true data set if desired and then returns the extrapolated data
            set.
                    
        """
        # To ensure that all parameters are present
        assert (len(params)==5 or len(params)==3)

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], coef0=params[3], gamma=params[4])

        # Add the known training data to the predicted points list
        y_return = y_tot[:training_dim].tolist()
        
        # While the length of the predicted points list is less than the total number of 
        # needed points
        while len(y_return) < len(y_tot):
            # Ensure that there are enough points the the predicted points list to be 
            # properly formatted.  Re-fitting the model only occurs when there are enough 
            # data points for the data to be properly formatted
            if len(y_return) % seq == 0:
                # Format the data
                X_train, y_train = time_series_data(y_return)
                # Fit the model
                kr.fit(X_train, y_train)
            # Predict the next point in the data set and add it to the list
            next_input = [[y_return[-2], y_return[-1]]]    
            next = kr.predict(next_input)    
            y_return.append(next[0])

        # Print the MSE error if needed
        if verbose:
            print ('KERNEL RIDGE CONTINUOUS RETRAIN MSE VALUE: ', mse(y_tot, y_return))

        # Return the predicted list
        return y_return, mse(y_tot, y_return)

    ##################################################
    #
    # COMPLETE DATA SET IS UNKNOWN
    #
    ##################################################

    #############################
    # UNKNOWN DATA 
    #############################
    def unknown_data (self, X_train, y_train, y_known, training_dim, params, verbose=True):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_tot (a list of numpy array): the total set of data points (training plus validation)
                training_dim (an int): the size of the traing data (i.e. the number of points
                    from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.
            Returns:
                y_return (a list): the known points and the extrapolated data points
                Unnamed (a float): the MSE error between the true data and the predicted
                    data
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set.  Prints the MSE 
            score of the extrapolated data set compared to the true data set if desired and
            then returns the extrapolated data set.
                    
        """
        # To ensure that all parameters are present 
        assert len(params)==5

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], 
            coef0=params[3], gamma=params[4])    

        # Fit the model to the training data
        kr.fit(X_train, y_train)

        # Use the trained model to predict the points in the validation set
        y_return = y_known
        next_input = [[y_return[-1]]]
        last = y_return[-1]
        while len(y_return) < len(y_tot):
            try:
                next = kr.predict(next_input)
            except:
                print ('Overflow encountered on predicton')
                return None, 1e10
            y_return.append(next[0])
            next_input =[[next[0]]]

        # Return the predicted points
        return y_return

    #############################
    # UNKNOWN DATA CR (CONTINUOUS RETRAIN)
    #############################
    def unknown_data_cr (self, X_train, y_train, y_known,
        training_dim, params, verbose):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_tot (a list of numpy array): the total set of data points
                training_dim (an int): the size of the traing data (i.e. the number of 
                    points from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.
                seq (an int): the length of the series to use in the time series formatting (default 
                    value is 2)    
            Returns:
                y_return (a list): the known points and the extrapolated data points
                Unnamed (a float): the MSE error between the true data and the predicted
                    data
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set. Kernel ridge 
            regression is performed after each point is extrapolated to hopefully decrease 
            the average MSE score.  Prints the MSE score of the extrapolated data set 
            compared to the true data set if desired and then returns the extrapolated data
            set.
                    
        """
        # To ensure that all parameters are present
        assert len(params)==5

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], coef0=params[3], gamma=params[4])

        # Add the known training data to the predicted points list
        y_return = y_known
        
        # While the length of the predicted points list is less than the total number of 
        # needed points
        while len(y_return) < len(y_tot):
            # Ensure that there are enough points the the predicted points list to be 
            # properly formatted.  Re-fitting the model only occurs when there are enough 
            # data points for the data to be properly formatted
            if len(y_return) % seq == 0:
                # Format the data
                X_train, y_train = time_series_data(y_return)
                # Fit the model
                kr.fit(X_train, y_train)
            # Predict the next point in the data set and add it to the list
            next_input = [[ y_return[-1]]]    
            next = kr.predict(next_input)    
            y_return.append(next[0])

        # Return the predicted list
        return y_return


    #############################
    # UNKNOWN DATA SEQ (SEQUENTIAL)
    #############################
    def unknown_data_seq (self, X_train, y_train, y_known, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_known (a list): the known data (aka the training data) in a 1D list
               total_points (an int): the total number of points needed in the final data set
                    including the number of training points
                training_dim (an int): the size of the traing data (i.e. the number of points
                    from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.
            Returns:
                y_return (a list): the known points and the extrapolated data points
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set. 
        """
        # To ensure that all parameters are present 
        assert len(params)==5

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], coef0=params[3], gamma=params[4])    

        # Fit the model to the training data
        kr.fit(X_train, y_train)


        if not isConvergence:
            # Use the trained model to predict the points in the validation set
            y_return = y_known
            next_input = [[y_return[-2], y_return[-1]]]
            last = y_return[-1]
            while len(y_return) < total_points:
                next = kr.predict(next_input)
                y_return.append(next[0])
                next_input =[[last, next[0]]]
                last = next[0]

        else:
            y_return = y_known.tolist()
            next_input = [[y_return[-2], y_return[-1]]]
            last = y_return[-1]
            second_last = y_return[-2]
            while np.abs(second_last - last) > convergence_threshold:
                next = kr.predict(next_input)
                y_return.append(next[0])
                next_input =[[last, next[0]]]
                second_last = last
                last = next[0]


        # Return the predicted points
        return y_return
    #############################
    #  UNKNOWN DATA CR SEQ (CONTINUOUS DATA, SEQUENTIAL)
    #############################
    def unknown_data_cr_seq (self, X_train, y_train, y_known, total_points, training_dim, params, isConvergence, convergence_threshold, verbose, seq=2):
        """
            Inputs:
                X_train (a list or numpy array): the x component of the training data
                y_train (a list or numpy array): the y component of the training data
                y_known (a list): the known data (aka the training data) in a 1D list
               total_points (an int): the total number of points needed in the final data set
                    including the number of training points
                training_dim (an int): the size of the traing data (i.e. the number of points
                    from y_tot that are used in the training)
                params (a list): contains the parameters of the kernel ridge regression 
                    algorithm.  In order: kernel, degree, alpha, coef0, gamma.
                verbose (a boolean): True case: prints the MSE score of the extrapolated data
                    when compared to the true data.
                seq (an int): the length of the series to use in the time series formatting (default 
                    value is 2)    
            Returns:
                y_return (a list): the known points and the extrapolated data points
            Performs kernel ridge regression on the given data set using the given parameters
            and then extrapolates data points to get a complete data set. Kernel ridge 
            regression is performed after each point is extrapolated to hopefully decrease 
            the average MSE score.  Prints the MSE score of the extrapolated data set 
            compared to the true data set if desired and then returns the extrapolated data
            set.
                    
        """
        # To ensure that all parameters are present
        assert len(params)==5

        # Set up the model
        kr = KernelRidge (kernel=params[0], degree=params[1], alpha=params[2], coef0=params[3], gamma=params[4])

        # Add the known training data to the predicted points list
        y_return = y_known.tolist()
        second_last = y_return[-2]
        last = y_return[-1]

        X_train, y_train = time_series_data(y_return)
        # Fit the model
        kr.fit(X_train, y_train)

        # While the length of the predicted points list is less than the total number of 
        # needed points
        while np.abs(second_last - last) > convergence_threshold:
            print (second_last, last)
            # Ensure that there are enough points the the predicted points list to be 
            # properly formatted.  Re-fitting the model only occurs when there are enough 
            # data points for the data to be properly formatted
            if len(y_return) % seq == 0:
                # Format the data
                X_train, y_train = time_series_data(y_return)
                # Fit the model
                kr.fit(X_train, y_train)
            # Predict the next point in the data set and add it to the list
            next_input = [[y_return[-2], y_return[-1]]]    
            next = kr.predict(next_input)
            second_last = last
            last = next    
            y_return.append(next[0])

        # Return the predicted list
        return y_return

    ##################################################
    #
    # SERIAL HYPERPARAMETER TUNING 
    #
    ##################################################

    ##############################
    # TUNE SERIAL REGULAR
    ##############################
    def tune_serial_regular (self, params_list, X_train, y_train, dim, y_tot,
        verbose=True, isReturnBest = True, threshold = 0):
        """
            Inputs:
                params_list
                X_train
                y_train
                training_dim
                verbose
                isReturnBest
            Returns:
                best_model
        """
        best_models = []    

        # Extract the parameter lists (i.e. the parameter values to be cycled through)
        kernels = params_list[0]
        degrees = params_list[1]
        alphas= params_list[2]
        coef0s = params_list[3]
        gammas = params_list[4]

        # Lists to store possible parameter combinations, first with only the polynomial
        # kernel and second with all other kernels
        params_list_polynomial = []
        params_list_other = []
        params_list_formatted = []
        # If the polynomial kerel is in the kernels list, create all possible 
        if 'polynomial' in kernels:
            params_list_polynomial = list(product(kernels, degrees, alphas, coef0s, gammas))
            kernels.remove('polynomial')
        params_list_other = list(product(kernels, alphas, gammas))

        if not params_list_polynomial:
            params_list_formatted = params_list_other
        elif not params_list_formatted:
            params_list_formatted = params_list_polynomial
        else:
            params_list_formatted = np.concatenate((params_list_polynomial, params_list_other))

        # Create a list of all possible parameter combinations
        params_list_formatted = list(product(kernels, degrees, alphas, coef0s, gammas))

        # Set up the variables to hold information about the best set of parameters
        best_score = 100
        best_model= []
        best_extrapolation = []

        # Loop through all possible combinations of parameters
        for params in params_list_formatted:
            # Perform the kernel ridge regression
            y_return, mse_err = self.known_data (X_train, y_train, y_tot, dim, params, verbose=False)
            # If the current model is best, make it the new best model
            if mse_err < best_score:
                best_score = mse_err
                best_model = params
                best_extrapolation = y_return
            if best_score < threshold:
                return best_model
            if best_score < threshold and isReturnBest:
                best_models.append(best_score)
                best_models.append(best_model)
                best_models.append(best_extrapolation)
                return best_models
        # If requested, print the best scores and parameters to the console
        if verbose:
            print ('BEST KERNEL RIDGE REGRESSION SCORE: ', best_score)
            print ('BEST KERNEL RIDGE REGRESSION PARAMETERS: ', best_model)

        # If requested, add the best scores, models, and extrapolations to a list and then
        # return the best parameters and the list
        if isReturnBest:
            best_models.append(best_score)
            best_models.append(best_model)
            best_models.append(best_extrapolation)
            return best_models

        # If isReturnBest is false, only return the best parameters
        return best_model

    ##############################
    # TUNE SERIAL SEQ (SEQUENTIAL)
    ##############################
    def tune_serial_seq (self, params_list, X_train, y_train, dim, y_tot,
        verbose=True, isReturnBest = True, threshold = 0):
        """
            Inputs:
                params_list
                X_train
                y_train
                training_dim
                verbose
                isReturnBest
            Returns:
                best_model
        """
        best_models = []    

        # Extract the parameter lists (i.e. the parameter values to be cycled through)
        kernels = params_list[0]
        degrees = params_list[1]
        alphas= params_list[2]
        coef0s = params_list[3]
        gammas = params_list[4]

        # Lists to store possible parameter combinations, first with only the polynomial
        # kernel and second with all other kernels
        params_list_polynomial = []
        params_list_other = []
        params_list_formatted = []
        # If the polynomial kerel is in the kernels list, create all possible 
        if 'polynomial' in kernels:
            params_list_polynomial = list(product(kernels, degrees, alphas, coef0s, gammas))
            kernels.remove('polynomial')
        params_list_other = list(product(kernels, alphas, gammas))

        if not params_list_polynomial:
            params_list_formatted = params_list_other
        elif not params_list_formatted:
            params_list_formatted = params_list_polynomial
        else:
            params_list_formatted = np.concatenate((params_list_polynomial, params_list_other))

        # Set up the variables to hold information about the best set of parameters
        best_score = 10e10
        best_model= []
        best_extrapolation = []

        # Loop through all possible combinations of parameters
        for params in params_list_formatted:
            # Perform the kernel ridge regression
            y_return, mse_err = self.known_data_seq (X_train, y_train, y_tot, dim, params, verbose=False)
            # If the current model is best, make it the new best model
            if mse_err < best_score:
                best_score = mse_err
                best_model = params
                best_extrapolation = y_return
            if best_score < threshold:
                return best_model
            if best_score < threshold and isReturnBest:
                best_models.append(best_score)
                best_models.append(best_model)
                best_models.append(best_extrapolation)
                return best_models                            
       
        # If requested, print the best scores and parameters to the console
        if verbose:
            print ('BEST KERNEL RIDGE REGRESSION SCORE: ', best_score)
            print ('BEST KERNEL RIDGE REGRESSION PARAMETERS: ', best_model)

        # If requested, add the best scores, models, and extrapolations to a list and then
        # return the best parameters and the list
        if isReturnBest:
            best_models.append(best_score)
            best_models.append(best_model)
            best_models.append(best_extrapolation)
            return best_models

        # If isReturnBest is false, only return the best parameters
        return best_model

    ##############################
    # TUNE SERIAL REGULAR CR (CONTINUOUS RETRAIN)
    ##############################
    def tune_serial_regular_cr (self, params_list, X_train, y_train, dim, y_tot,
        verbose=True, isReturnBest = True, threshold = 0):
        """
            Inputs:
                params_list
                X_train
                y_train
                training_dim
                verbose
                isReturnBest
            Returns:
                best_model
        """
        best_models = []    

        # Extract the parameter lists (i.e. the parameter values to be cycled through)
        kernels = params_list[0]
        degrees = params_list[1]
        alphas= params_list[2]
        coef0s = params_list[3]
        gammas = params_list[4]

        # Lists to store possible parameter combinations, first with only the polynomial
        # kernel and second with all other kernels
        params_list_polynomial = []
        params_list_other = []
        params_list_formatted = []
        # If the polynomial kerel is in the kernels list, create all possible 
        if 'polynomial' in kernels:
            params_list_polynomial = list(product(kernels, degrees, alphas, coef0s, gammas))
            kernels.remove('polynomial')
        params_list_other = list(product(kernels, alphas, gammas))

        if not params_list_polynomial:
            params_list_formatted = params_list_other
        elif not params_list_formatted:
            params_list_formatted = params_list_polynomial
        else:
            params_list_formatted = np.concatenate((params_list_polynomial, params_list_other))

        # Create a list of all possible parameter combinations
        params_list_formatted = list(product(kernels, degrees, alphas, coef0s, gammas))

        # Set up the variables to hold information about the best set of parameters
        best_score = 100
        best_model= []
        best_extrapolation = []

        # Loop through all possible combinations of parameters
        for params in params_list_formatted:
            # Perform the kernel ridge regression
            y_return, mse_err = self.known_data_cr (X_train, y_train, y_tot, dim, params, verbose=False)
            # If the current model is best, make it the new best model
            if mse_err < best_score:
                best_score = mse_err
                best_model = params
                best_extrapolation = y_return
            if best_score < threshold:
                return best_model
            if best_score < threshold and isReturnBest:
                best_models.append(best_score)
                best_models.append(best_model)
                best_models.append(best_extrapolation)
                return best_models
        # If requested, print the best scores and parameters to the console
        if verbose:
            print ('BEST KERNEL RIDGE REGRESSION SCORE: ', best_score)
            print ('BEST KERNEL RIDGE REGRESSION PARAMETERS: ', best_model)

        # If requested, add the best scores, models, and extrapolations to a list and then
        # return the best parameters and the list
        if isReturnBest:
            best_models.append(best_score)
            best_models.append(best_model)
            best_models.append(best_extrapolation)
            return best_models

        # If isReturnBest is false, only return the best parameters
        return best_model

    ##############################
    # TUNE SERIAL SEQ CR (SEQUENTIAL, CONTINUOUS RETRAIN) 
    ##############################
    def tune_serial_seq_cr (self, params_list, X_train, y_train, dim, y_tot,
        verbose=True, isReturnBest = True, threshold = 0):
        """
            Inputs:
                params_list
                X_train
                y_train
                training_dim
                verbose
                isReturnBest
            Returns:
                best_model
        """
        best_models = []    

        # Extract the parameter lists (i.e. the parameter values to be cycled through)
        kernels = params_list[0]
        degrees = params_list[1]
        alphas= params_list[2]
        coef0s = params_list[3]
        gammas = params_list[4]

        # Lists to store possible parameter combinations, first with only the polynomial
        # kernel and second with all other kernels
        params_list_polynomial = []
        params_list_other = []
        params_list_formatted = []
        # If the polynomial kerel is in the kernels list, create all possible 
        if 'polynomial' in kernels:
            params_list_polynomial = list(product(kernels, degrees, alphas, coef0s, gammas))
            kernels.remove('polynomial')
        params_list_other = list(product(kernels, alphas, gammas))

        if not params_list_polynomial:
            params_list_formatted = params_list_other
        elif not params_list_formatted:
            params_list_formatted = params_list_polynomial
        else:
            params_list_formatted = np.concatenate((params_list_polynomial, params_list_other))

        # Set up the variables to hold information about the best set of parameters
        best_score = 100
        best_model= []
        best_extrapolation = []

        # Loop through all possible combinations of parameters
        for params in params_list_formatted:
            # Perform the kernel ridge regression
            y_return, mse_err = self.known_data_cr_ts (X_train, y_train, y_tot, dim, params, verbose=False)
            # If the current model is best, make it the new best model
            if mse_err < best_score:
                best_score = mse_err
                best_model = params
                best_extrapolation = y_return
            if best_score < threshold:
                return best_model
            if best_score < threshold and isReturnBest:
                best_models.append(best_score)
                best_models.append(best_model)
                best_models.append(best_extrapolation)
                return best_models                            
       
        # If requested, print the best scores and parameters to the console
        if verbose:
            print ('BEST KERNEL RIDGE REGRESSION SCORE: ', best_score)
            print ('BEST KERNEL RIDGE REGRESSION PARAMETERS: ', best_model)

        # If requested, add the best scores, models, and extrapolations to a list and then
        # return the best parameters and the list
        if isReturnBest:
            best_models.append(best_score)
            best_models.append(best_model)
            best_models.append(best_extrapolation)
            return best_models

        # If isReturnBest is false, only return the best parameters
        return best_model        

    ##################################################
    #
    # PARALLEL HYPERPARAMETER TUNING 
    #
    ##################################################


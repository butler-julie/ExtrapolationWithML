########################################
# Convergence
# Julie Butler Hartley
# Date Created: October 22, 2020
# Last Modified: October 22, 2020
# Version 1.0.0
#
# Extrapolates a data set to a converged value using either linear regression, ridge 
# regresion, or kernel ridge regression.  Take a list of optimal hyperparameters or
# can perform hyperparameter tuning.
########################################


#############################
# IMPORTS
#############################
# THIRD-PARTY IMPORTS
# For array handling
import numpy as np

# LOCAL IMPORTS
# Linear regression codes
from LinearRegression import LinearRegressionAnalysis
# Ridge regression codes
from RidgeRegression import RidgeRegressionAnalysis
# Kernel ridge regression codes
from KernelRidgeRegression import KernelRidgeRegressionAnalysis
# Support methods, including graphing capabilities
from RegressionSupport import *
# Data sets (mostly physics related)
from DataSets import *
from ElectronGas import *

isLinearRegression = False
isRidgeRegression = False
isKernelRidgeRegression = False

isOptimalParameters = True
params_lr = [True, True]
params_rr = [True, 2.0165510387247442e-48, 'saga']
params_krr = []

isConvergence = True
convergence_threshold = 1e-5

isAutoRegression = True

name, dim, X_tot, y_tot = rs_1_N_26()
dim = 30

X_sequential, y_sequential = time_series_data (y_tot)
total_points = 100

if isOptimalParameters:
    if isLinearRegression:
        lr = LinearRegressionAnalysis
        if isAutoRegression: 
            y_return = lr.unknown_data_cr_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
        else:
            y_return = lr.unknown_data_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
    if isRidgeRegression:
        rr = RidgeRegressionAnalysis
        if isAutoRegression: 
            y_return = rr.unknown_data_cr_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
        else:
            y_return = rr.unknown_data_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
    if isKernelRidgeRegression:
        krr = KernelRidgeRegressionAnalysis
        if isAutoRegression: 
            y_return = krr.unknown_data_cr_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
        else:
            y_return = krr.unknown_data_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
else:
    if isLinearRegression:
        lr = LinearRegressionAnalysis
        params_list = [[True, False], [True, False]]
        params_lr = lr.tune_serial_seq (params_list, X_sequential, y_sequential, dim, y_tot, verbose=True, isReturnBest = False, threshold = 0)
        if isAutoRegression: 
            y_return = lr.unknown_data_cr_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params_lr, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
        else:
            y_return = lr.unknown_data_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params_lr, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
    if isRidgeRegression:
        rr = RidgeRegressionAnalysis
        normalizes = [True, False]
        solvers= ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        alphas = np.logspace(-50, 0, 500)
        params_list = [normalizes, alphas, solvers]
        params_rr = rr.tune_serial_seq (params_list, X_sequential, y_sequential, dim, y_tot, verbose=True, isReturnBest = False, threshold = 0)
        if isAutoRegression: 
            y_return = rr.unknown_data_cr_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params_rr, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
        else:
            y_return = rr.unknown_data_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params_rr, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
    if isKernelRidgeRegression:
        krr = KernelRidgeRegressionAnalysis
        kernels = ['polynomial', 'sigmoid']
        degrees = [1, 2, 3, 4]
        alphas = np.logspace(-20, -10, 20)
        coef0s = np.arange(-5, 6, 1)
        gammas = np.logspace(0, 2, 15)
        params_list = [kernels, degrees, alphas, coef0s, gammas]
        params_krr = krr.tune_serial_seq (params_list, X_sequential, y_sequential, dim, y_tot, verbose=True, isReturnBest = False, threshold = 0)
        if isAutoRegression: 
            y_return = krr.unknown_data_cr_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params_krr, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))
        else:
            y_return = krr.unknown_data_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params_krr, isConvergence, convergence_threshold, verbose=True)
            print ('CONVERGED??', y_return[-1])
            print ('LENGTH??', len(y_return))

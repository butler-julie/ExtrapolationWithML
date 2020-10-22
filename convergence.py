#Train only on "converged" region?
#Fit to exponentials and test that as well


#############################
# IMPORTS
#############################
# SYSTEM LEVEL IMPORTS
# Import files from other directories
import sys

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
# Changing the import directory
sys.path.append('../../DataSets/')
# Data sets (mostly physics related)
from DataSets import *
from ElectronGas import *

#X_train = np.arange(1, 51)
#y_train = 1/X_train
#y_tot = 1/X_train
#total_points = 100
training_dim = 50
#dim = len(X_train)
#X_tot = np.arange(1, 101)
#y_tot = 1/X_tot
params = [True, 2.0165510387247442e-48, 'saga']
isConvergence = True
convergence_threshold = 1e-5
#X_sequential, y_sequential = time_series_data (y_train)


name, dim, X_tot, y_tot = rs_1_N_26()
dim = 30

# rs = 1
#X_tot = np.array([10, 26, 42, 74, 114, 122, 138, 178])
#dim = 6
#y_tot = np.array([-0.1985319074, -0.2023078493, -0.188448089, -0.2068900836, -0.1969145017, -0.1954344112, -0.1945482018, -0.206784843])

# rs = 0.5
#X_tot = ([10, 74, 98, 114, 138, 198])
#dim = 4
#y_tot = np.array([-0.247735287, -0.2514914497, -0.2572199544, -0.2526152815, -0.2491438612, -0.2521939083])

# rs=2
#X_tot = ([10, 74, 114, 138, 198])
#dim = 4
#y_tot = np.array([-0.1416735733, -0.1398474626, -0.1394580773, -0.1399823693, -0.1477833918])


#import matplotlib.pyplot as plt 
#plt.scatter(X_tot, y_tot)
#plt.show()

X_sequential, y_sequential = time_series_data (y_tot)
total_points = 100
rr = RidgeRegressionAnalysis()

#y_return = rr.unknown_data_cr_seq (X_sequential, y_sequential, y_tot, total_points, training_dim, params, isConvergence, convergence_threshold, verbose=True)


normalizes = [True, False]
solvers= ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
alphas = np.logspace(-50, 0, 500)
params_list = [normalizes, alphas, solvers]
#params_list = [[True, False], [True, False]]
#kernels = ['polynomial', 'sigmoid']
#degrees = [1, 2, 3, 4]
#alphas = np.logspace(-20, -10, 20)
#coef0s = np.arange(-5, 6, 1)
#gammas = np.logspace(0, 2, 15)
#params_list = [kernels, degrees, alphas, coef0s, gammas]


rr.tune_serial_seq (params_list, X_sequential, y_sequential, dim, y_tot,
        verbose=True, isReturnBest = True, threshold = 0)


print ('CONVERGED??', y_return[-1])
print ('LENGTH??', len(y_return))
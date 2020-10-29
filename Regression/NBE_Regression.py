#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############################
# IMPORTS
#############################
# SYSTEM LEVEL IMPORTS
# Import files from other directories
import sys

# THIRD-PARTY IMPORTS
# For array handling
import numpy as np
#import matplotlib.pyplot as plt

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
sys.path.append('../DataSets/')
# Data sets (mostly physics related)
from DataSets import *
from ElectronGas import *
from NuclearBindingEnergy import *
from EquationOfState import *


# In[2]:


X_tot, y_tot, design_matrix = EquationOfState()
print(len(X_tot))
training_dim = 45


# In[3]:


X_train, y_train = time_series_data (y_tot[:training_dim], 2)


# In[ ]:





# In[4]:


#############################
# LINEAR REGRESSION PARAMETERS
#############################
# Possible values of parameters for the linear regression algorithm
params_list_lr = [[True, False], [True, False]]


#############################
# RIDGE REGRESSION PARAMETERS
#############################
# Possible values of parameters for the ridge regression algorithm
normalizes = [True, False]
solvers= ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
alphas = np.logspace(-50, 0, 500)
params_list_rr = [normalizes, alphas, solvers]

#############################
# KERNEL RIDGE REGRESSION PARAMETERS
#############################
# Possible values of parameters for the kernel ridge regression algorithm
kernels = ['polynomial']
degrees = [4/3]
alphas = np.logspace(-5, 0, 150)
coef0s = [2, 3, 4]
gammas = np.logspace(-5, 0, 150)
params_list_krr = [kernels, degrees, alphas, coef0s, gammas]


# In[5]:


lr = KernelRidgeRegressionAnalysis()


# In[ ]:


best_models = lr.tune_serial_seq (params_list_krr, X_train, y_train, training_dim, y_tot,
        verbose=True, isReturnBest = True, threshold = 0)
# Save the best extrapolated data set and extrapolated error to variables
# to be used later
data = best_models[2]
score = best_models[0]


# In[ ]:


print('****************************************')
print('BEST MSE: ', score)
print('****************************************')


# In[ ]:

print('****************************************')
print('BEST PARAMETERS: ', best_models[1])
print('****************************************')


# In[ ]:


# In[ ]:

print('****************************************')
print('RATIO SCORE: ', score/y_tot[-1])
print('****************************************')



# In[ ]:


#80 : ('polynomial', 1.3333333333333333, 0.029470517025518096, 3, 0.0003727593720314938)
#45 : 'polynomial', 1.3333333333333333, 0.0005428675439323859, 2, 0.0035564803062231283


# In[ ]:


#ratio score: 0.002371279714439891 (80)
#ratio score: 0.0021172180314645976 (45)    


# In[ ]:


#params = ['polynomial', 4/3, 0.02947, 3.01, 0.0003727]
#training_dim = 80
#X_train, y_train = time_series_data (y_tot[:training_dim], 2)#

#y_new, mse_new = lr.known_data_seq (X_train, y_train, y_tot, training_dim, params, verbose=True, seq=2)

#print(mse_new/y_tot[-1])

#plt.plot(X_tot, y_new)
#plt.plot(X_tot, y_tot)


# In[ ]:


X_test, y_test = time_series_data (y_tot, 1)

X_test = X_test.flatten()


# In[ ]:


plt.plot(X_test, y_test)


# In[ ]:





#################################################
# Simple Regression Example
# Julie Butler Hartley
# Version 1.0.1
# Date Created: October 28, 2020
# Last Modified: October 28, 2020
#

#################################################

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
sys.path.append('../DataSets/')
# Data sets (mostly physics related)
from DataSets import *
from ElectronGas import *
from NuclearBindingEnergy import *

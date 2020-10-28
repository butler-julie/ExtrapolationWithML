##################################################
# Equation of State
# Julie Butler Hartley
# Date Created: October 28, 2020
# Last Modified: October 28, 2020
# Version 1.0.0
#
# Contains a function that imports the data from the file EoS.csv and returns
# it in formatted numpy arrays for density, energies, and a design matrix for
# the equation of state.  This is for dense nuclear matter
# Based of the code written by Morten Hjorth-Jensen which is located here:
# https://compphysics.github.io/MachineLearning/doc/pub/week35/html/week35.html
##################################################

##############################
# IMPORTS
##############################
# THIRD-PARTY IMPORTS
# For arrays
import numpy as np
# For importing and formatting the data file 
import pandas as pd

# EQUATION OF STATE
def EquationOfState():
    # Open the data file
    infile = open("EoS.csv",'r')

    # Read the EoS data as  csv file and organize the data into two arrays with density and energies
    EoS = pd.read_csv(infile, names=('Density', 'Energy'))
    EoS['Energy'] = pd.to_numeric(EoS['Energy'], errors='coerce')
    EoS = EoS.dropna()
    Energies = EoS['Energy']
    Density = EoS['Density']
    
    #  The design matrix now as function of various polytrops
    X = np.zeros((len(Density),4))
    X[:,3] = Density**(4.0/3.0)
    X[:,2] = Density
    X[:,1] = Density**(2.0/3.0)
    X[:,0] = 1
    
    # Returns the densities and energies as numpy arrays as well as the design matrix
    return np.asarray(Density), np.asarray(Energies), X

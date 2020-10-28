##################################################
# Nuclear Binding Energy
# Julie Butler Hartley
# Date Created: October 27, 2020
# Last Modifed: October 27, 2020
# Version 1.0.0
#
# A function for importing the data from the file MassEval2016.dat, which contains binding energies of all of
# the isotopes.  The function returns the number of nucleons, the binding energies, and a design matrix based
# off of fitting an equation to the binding energies as a function of the number of nucleons.
# 
# Code based off of a code written by Morten Hjorth-Jensen located here:
# https://compphysics.github.io/MachineLearning/doc/pub/week34/html/week34.html
##################################################

##############################
# IMPORTS
##############################
# For arrays
import numpy as np
# For reading and formatting the data file
import pandas as pd

# NUCLEAR BINDING ENERGY
def NuclearBindingEnergy ():
    """
        Inputs:
            None.
        Returns:
            A (a numpy array): the number of nucleons per data point
            Energies (a numpy array): the corresponding binding energies
            X (a 2D numpy array): A design matrix based off of the equation relating the number of nucleons
                and the binding energies
        Reads from file the number of nucleons and binding energies for isotopes.  Returns the data in numpy arrays
        as well as a design matrix for the equation relating the number of nucleons and the binding energies
    """                                                                                                                         
    # This is taken from the data file of the mass 2016 evaluation.                                                               
    # All files are 3436 lines long with 124 character per line.                                                                  
           # Headers are 39 lines long.                                                                                           
       # col 1     :  Fortran character control: 1 = page feed  0 = line feed                                                     
       # format    :  a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.5                     
       # These formats are reflected in the pandas widths variable below, see the statement                                       
       # widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),                                                            
       # Pandas has also a variable header, with length 39 in this case.                                                          

    # Open the data file
    infile = open("MassEval2016.dat",'r')

    # Read the experimental data with Pandas
    Masses = pd.read_fwf(infile, usecols=(2,3,4,6,11),
                  names=('N', 'Z', 'A', 'Element', 'Ebinding'),
                  widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
                  header=39,
                  index_col=False)

    # Extrapolated values are indicated by '#' in place of the decimal place, so
    # the Ebinding column won't be numeric. Coerce to float and drop these entries.
    Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce')
    Masses = Masses.dropna()
    # Convert from keV to MeV.
    Masses['Ebinding'] /= 1000

    # Group the DataFrame by nucleon number, A.
    Masses = Masses.groupby('A')
    # Find the rows of the grouped DataFrame with the maximum binding energy.
    Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])

    # Separate the relevant data
    A = Masses['A']
    Z = Masses['Z']
    N = Masses['N']
    Element = Masses['Element']
    Energies = Masses['Ebinding']

    # Now we set up the design matrix X
    X = np.zeros((len(A),5))
    X[:,0] = 1
    X[:,1] = A
    X[:,2] = A**(2.0/3.0)
    X[:,3] = A**(-1.0/3.0)
    X[:,4] = A**(-1.0)
    
    return np.asarray(A), np.asarray(Energies), X

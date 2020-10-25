#################################################
# Data Sets
# Julie Hartley
# Version 1.1.0
# Created: December 28, 2019
# Last Modified: October 25, 2020
#
# Collection of data sets to be used in the many-body/machine learning extrapolation 
# project.  Also includes some non-physical data sets in the form of polynomials and
# trigonometric functions.
#################################################

#################################################
# OUTLINE:
# Note that these are not the names of the methods but are descriptive titles
# Coupled Cluster Data Sets:
#   Pairing Model:
#       Vary Dimension
#       Vary Interaction Positive
#       Vary Interaction Negative
#   Infinite Matter:
#       Density Energy Nmax 20 Number of Particles 54
# In-Medium Similarity Renormalization Group Data Sets:
#   Paring Model (4p4h):
#       IMSRG Energies
# Similarity Renormalization Group Data Sets:
#   Pairing Model (4p4h):
#       Full Hamiltonian Evolution
#       Ground State Energy Evolution
# Generated Function Data Sets
#   Linear
#   Polynomial of Nth degree
#   Sine
#   Cosine
#   Tangent
#   Arctangent
#################################################

#################################################
# Note on the generated function data sets:
# Unlike the other data sets, these functions accept arguments, which differ
# per data set and are defined below in the function docstring.  The x range 
# for the function is from -1 to 1, containing 200 points separated by 0.01.
# the training dimension is set to be the first 50 points (which would be the 
# x range between -1 and -0.5).
#################################################

#############################
# IMPORTS
#############################
# THIRD PARTY IMPORTS
# for creating ranges and arrays
import numpy as np

#################################################
# COUPLED CLUSTER
#################################################

#############################
# PAIRING MODEL
#############################
# VARYDIMENSION 
def VaryDimension ():
    """
        Inputs:
            None.
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Origin:
            X component is the number of particles/holes in the pairing model, y 
            component are the corresponding coupled cluster doubles correlation energies.
            Data set created using the code found at https://nucleartalent.github.io/
            ManyBody2018/doc/pub/CCM/html/CCM.html.     
    """
    data_name ='VaryDimension'
    training_dim = 12
    X_tot = np.arange(2, 42, 2)
    y_tot = np.array([-0.03077640549, -0.08336233266, -0.1446729567, -0.2116753732, -0.2830637392,
                      -0.3581341341, -0.436462435, -0.5177783846, -0.6019067271, -0.6887363571, 
                      -0.7782028952, -0.8702784034, -0.9649652536, -1.062292565, -1.16231451, 
    	              -1.265109911, -1.370782966, -1.479465113, -1.591317992, -1.70653767])
    return data_name, training_dim, X_tot, y_tot 

# VARYINTERACTIONNEGATIVE
def VaryInteractionNegative ():
    """
        Inputs:
            None.
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Origin:
            X component is the strength of the interaction between particles in the 
            pairing model (only negative values), y component are the corresponding
            coupled cluster doubles correlation energies. Data set created using the
            code found at https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html
            /CCM.html.     
    """
    data_name='VaryInteractionNegative'
    training_dim = 12
    X_tot = np.arange(-1, 0, 0.05)
    y_tot = np.array([-1.019822621,-0.9373428759,-0.8571531335,-0.7793624503,-0.7040887974,
        -0.6314601306,-0.561615627,-0.4947071038,-0.4309007163,-0.3703789126,-0.3133427645,
        -0.2600147228,-0.2106419338,-0.1655002064,-0.1248988336,-0.08918647296,-0.05875839719,
        -0.03406548992,-0.01562553455,-0.004037522178])
    return data_name, training_dim, X_tot, y_tot

# VARYINTERACTIONPOSITIVE
def VaryInteractionPositive ():
    """
        Inputs:
            None.
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Origin:
            X component is the strength of the interaction between particles in the 
            pairing model (only positive values), y component are the corresponding
            coupled cluster doubles correlation energies. Data set created using the
            code found at https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html
            /CCM.html.     
    """
    data_name = 'VaryInteractionPositive'
    training_dim = 10
    X_tot = np.arange(0.05, 0.85, 0.05)
    y_tot = np.array([-0.004334904077,-0.01801896484,-0.04222576507,-0.07838310563,-0.128252924,
        -0.1940453966,-0.2785866456,-0.3855739487,-0.5199809785,-0.6887363571,-0.9019400869,-1.175251697,
        -1.535217909,-2.033720441,-2.80365727,-4.719209688])
    return data_name, training_dim, X_tot, y_tot

# VARYINTERACTION
def VaryInteraction ():
    """
        Inputs:
            None.
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Origin:
            X component is the strength of the interaction between particles in the 
            pairing model, y component are the corresponding
            coupled cluster doubles correlation energies. Data set created using the
            code found at https://nucleartalent.github.io/ManyBody2018/doc/pub/CCM/html
            /CCM.html.      
    """
    data_name='VaryInteractionNegative'
    training_dim = 24
    X_tot = np.arange(-1, 0.85, 0.05)
    y_tot = np.array([-1.019822621,-0.9373428759,-0.8571531335,-0.7793624503,-0.7040887974,
        -0.6314601306,-0.561615627,-0.4947071038,-0.4309007163,-0.3703789126,-0.3133427645,
        -0.2600147228,-0.2106419338,-0.1655002064,-0.1248988336,-0.08918647296,-0.05875839719,
        -0.03406548992,-0.01562553455,-0.004037522178, -0.004334904077,-0.01801896484,-0.04222576507,
        -0.07838310563,-0.128252924,
        -0.1940453966,-0.2785866456,-0.3855739487,-0.5199809785,-0.6887363571,-0.9019400869,-1.175251697,
        -1.535217909,-2.033720441,-2.80365727,-4.719209688])
    return data_name, training_dim, X_tot, y_tot

#############################
# INFINITE NUCLEAR MATTER
#############################
# INFINITEMATTERDENSITYENERGYNMAX20NUMPART54
def InfiniteMatterDensityEnergyNmax20NumPart54 ():
    """
    Inputs:
        None.
    Returns:
        data_name (a string): the representative name of the data set
        training_dim (an int): the suggested training size
        X_tot (a numpy array): the x component of the data set
        y_tot (a numpy array): the y component of the data set
    Origin:
        X component is the density of the particles, y component are the 
        corresponding coupled cluster energies. Data set created using the
        code found at the Lecture Notes in Physics 936 GitHub.     
    """
    data_name = 'InfiniteMatterDensityEnergyNmax20NumPart54'    
    training_dim = 6
    X_tot = np.arange(0.025, 0.225, 0.025)
    y_tot = np.array([4.824946, 6.851108, 8.241808, 9.298385, 10.12724, 10.77886, 11.28275, 11.6588])
    return data_name, training_dim, X_tot, y_tot

#################################################
# IN-MEDIUM SIMILARITY RENORMALIZATION GROUP
#################################################
# IMSRGENERGIES
def IMSRGEnergies ():
    """
        Inputs:
            None.
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Origin:
            X component is the strength of the interaction between particles in the 
            pairing model, y component are the corresponding converged IMSRG energy. Data
            set created using code from Jane Kim.       
    """
    data_name = 'IMSRG'
    training_dim = 12
    X_tot = np.arange(0.05, 1.05, 0.05)
    y_tot = np.array([1.9492602, 1.8969958, 1.8431346, 1.7875963, 1.7302919, 1.6711222,
        1.6099759, 1.5467276, 
        1.4812354, 1.4133373, 1.3428474, 1.2695500, 1.1931931, 
        1.1134788, 1.0300502, 0.9424741, 0.8502147, 0.7572597, 0.6487456, 0.5374973])
    return data_name, training_dim, X_tot, y_tot

#################################################
# GENERATED FUNCTION DATA SETS
#################################################
# LINEAR
def linear (scale_factor, intercept):
    """
        Inputs:
            scale_factor (an int or float): the slope of the linear data set
            intercept (an int or float): the intercept of the linear data set
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Creates a linear data set that has the following form:
        y = scale_factor * x + intercept
    """    
    date_name = 'Linear'
    training_dim = 50
    X_tot = np.arange(-1, 1, 0.01)
    y_tot = scale_factor*X_tot + intercept
    return data_name, training_dim, X_tot, y_tot

# POLYNOMIAL
def polynomial (scale_factor, constant, degree):
    """
        Inputs:
            scale_factor (an int or float): A constant by which the polynomial data set
                is scaled
            constant (an int or float): a constant to be added to the x data
                before the degree and scale factor are applied
            degree (an int or float): the degree of the polynomial
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Creates a linear data set that has the following form:
        y = scale_factor * (x + constant)^degree
    """     
    date_name = 'Polynomial'
    training_dim = 50
    X_tot = np.arange(-1, 1, 0.01)
    y_tot = scale_factor*(X_tot + constant)**degree
    return data_name, training_dim, X_tot, y_tot

# SINE
def sine (scale_factor1, scale_factor2):
    """
        Inputs:
            scale_factor1 (an int or float): the scale fact to be applied to the data after the
                sine function is applied        
            scale_factor2 (an int or float): the scale factor to be applied to the x data before
                the sine function is applied
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Creates a linear data set that has the following form:
        y = scale_factor1 * sin(scalefactor2 * x)
    """     
    date_name = 'Sine'
    training_dim = 50
    X_tot = np.arange(-1, 1, 0.01)
    y_tot = scale_factor1*np.sin(scale_factor2*X_tot)
    return data_name, training_dim, X_tot, y_tot
    
# COSINE
def cosine (scale_factor1, scale_factor2):
    """
        Inputs:
            scale_factor1 (an int or float): the scale fact to be applied to the data after the
                cosine function is applied        
            scale_factor2 (an int or float): the scale factor to be applied to the x data before
                the cosine function is applied
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Creates a linear data set that has the following form:
        y = scale_factor1 * cos(scalefactor2 * x)
    """     
    date_name = 'Cosine'
    training_dim = 50
    X_tot = np.arange(-1, 1, 0.01)
    y_tot = scale_factor1*np.cos(scale_factor2*X_tot)
    return data_name, training_dim, X_tot, y_tot

# TAGNGENT
def tangent (scale_factor1, scale_factor2):
    """
        Inputs:
            scale_factor1 (an int or float): the scale fact to be applied to the data after the
                tangent function is applied        
            scale_factor2 (an int or float): the scale factor to be applied to the x data before
                the tangent function is applied
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Creates a linear data set that has the following form:
        y = scale_factor1 * tan(scalefactor2 * x)
    """     
    date_name = 'Tangent'
    training_dim = 50
    X_tot = np.arange(-1, 1, 0.01)
    y_tot = scale_factor1*np.tan(scale_factor2*X_tot)
    return data_name, training_dim, X_tot, y_tot

# ARCTANGENT
def linear (scale_factor1, scale_factor2):
    """
        Inputs:
            scale_factor1 (an int or float): the scale fact to be applied to the data after the
                arctangent function is applied        
            scale_factor2 (an int or float): the scale factor to be applied to the x data before
                the artangent function is applied
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Creates a linear data set that has the following form:
        y = scale_factor1 * arctan(scalefactor2 * x)
    """     
    date_name = 'Arctangent'
    training_dim = 50
    X_tot = np.arange(-1, 1, 0.01)
    y_tot = scale_factor1*np.arctan(scale_factor2*X_tot)
    return data_name, training_dim, X_tot, y_tot
    




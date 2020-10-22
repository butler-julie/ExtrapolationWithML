#################################################
# Data Sets
# Julie Hartley
# Version 1.1.0
# Created: December 28, 2019
# Last Modified: December 29, 2019
#
# Collection of data sets to be used in the many-body/machine learning extrapolation 
# project
#################################################

#################################################
# OUTLINE:
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
#################################################

#############################
# IMPORTS
#############################
# THIRD PARTY IMPORTS
# for creating ranges
import numpy as np

#############################
# CODES
#############################

#################################################
# COUPLED CLUSTER
#################################################

#############################
# VARY DIMENSION
#############################
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
    y_tot = np.array([-0.03077640549, -0.08336233266, -0.1446729567, -0.2116753732, -0.2830637392, -0.3581341341, -0.436462435, -0.5177783846,
    	-0.6019067271, -0.6887363571, -0.7782028952, -0.8702784034, -0.9649652536, -1.062292565, -1.16231451, 
    	-1.265109911, -1.370782966, -1.479465113, -1.591317992, -1.70653767])
    return data_name, training_dim, X_tot, y_tot 

#############################
# VARY INTERACTION NEGATIVE
#############################
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

#############################
# VARY INTERACTION POSITIVE
#############################
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

#############################
# Vary Interaction
#############################
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
        -0.03406548992,-0.01562553455,-0.004037522178, -0.004334904077,-0.01801896484,-0.04222576507,-0.07838310563,-0.128252924,
        -0.1940453966,-0.2785866456,-0.3855739487,-0.5199809785,-0.6887363571,-0.9019400869,-1.175251697,
        -1.535217909,-2.033720441,-2.80365727,-4.719209688])
    return data_name, training_dim, X_tot, y_tot

#############################
# INFINITE MATTER DENSITY ENERGY NMAX 20 NUM PART 54
#############################
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

#############################
# IMSRG ENERGIES
#############################
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
# SIMILARITY RENORMALIZATION GROUP
#################################################

#############################
# SRG FULL 
#############################
def SRGFull ():
    """
        Inputs:
            None.
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Origin:
            X component is SRG flow parameter (s), y component are the Hamiltonians at 
            that particular evolution of the flow parmater. Data set created using the
            SRG code written by author, found at ____________________ .
            
    """
    data_name = 'SRGFull'
    training_dim = 500
    X_tot = np.arange(0, 4.00, 0.001)
    y_tot = np.load('H.npy')[0:4000]
    return data_name, training_dim, X_tot, y_tot

#############################
# SRG GROUND STATES
#############################
def SRGGroundStates ():
    """
        Inputs:
            None.
        Returns:
            data_name (a string): the representative name of the data set
            training_dim (an int): the suggested training size
            X_tot (a numpy array): the x component of the data set
            y_tot (a numpy array): the y component of the data set
        Origin:
            X component is SRG flow parameter (s), y component are the ground state 
            energy at that particular evolution of the flow parmater. Data set created 
            using the SRG code written by author, found at ____________________ .
            
    """
    data_name = 'SRGGroundStates'
    training_dim = 500
    X_tot = np.arange(0, 2.00, 0.001)
    y_tot = np.load('H.npy')[0:2000]
    y_tot = np.array([y_tot[i][0] for i in range (len(y_tot))])
    return data_name, training_dim, X_tot, y_tot


def TwoDimElectronGasN26 ():
    """
    Inputs:
        None.
    Returns:
        data_name (a string): the representative name of the data set
        training_dim (an int): the suggested training size
        X_tot (a numpy array): the x component of the data set
        y_tot (a numpy array): the y component of the data set
    Origin:
        X component is SRG flow parameter (s), y component are the ground state 
        energy at that particular evolution of the flow parmater. Data set created 
        using the SRG code written by author, found at ____________________ .    
    """
    data_name = 'TwoDimElectronGasN26'
    training_dim = 16
    X_tot = np.array([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 50, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    y_tot = np.array([-0.67039E-01, -0.11728E+00, -0.14399E+00, -0.16185E+00, -0.17787E+00, -0.18554E+00, -0.18859E+00, -0.19101E+00, -0.19336E+00, -0.19458E+00, -0.19616E+00, -0.19721E+00, -0.19776E+00, -0.19876E+00, -0.19913E+00, -0.19959E+00, -0.19988E+00, -0.20039E+00, -0.20068E+00, -0.20153E+00, -0.20229E+00, -0.20309E+00, -0.20354E+00, -0.20381E+00, -0.20400E+00, -0.20414E+00, -0.20424E+00, -0.20431E+00, -0.20437E+00, -0.20442E+00, -0.20447E+00, -0.20450E+00, -0.20453E+00])
    return data_name, training_dim, X_tot, y_tot





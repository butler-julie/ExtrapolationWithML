##################################################
# Electron Gas 
# Julie Butler Hartley 
# Date Created: October 22, 2020
# Last Modified: October 23, 2020
# Version 1.0.0
#
# Data generated from applying couple cluster theory to an infinite electron gas.  Data 
# supplied by Morten Hjorth-Jensen.  Each method contains the number of orbitals (M) as a 
# function of correlations energy at various values of rs and N.  Note that these data 
# sets differ from the data sets located in DataSets.py by the addition on an addition
# return value, N.  Also note that many of these data sets are quite small, which can 
# make machine learning challenging.
#
# October 2020: This data is currently being used with various machine learning methods
# to attempt to recreate the results found by James Shepherd using traditional 
# extrapolation methods.
# https://arxiv.org/pdf/1605.05699.pdf
#
# Further information on infinite electron gases and couple cluster theory can be found at 
# the following links:
#   https://www.duo.uio.no/bitstream/handle/10852/41025/PhD-Baardsen.pdf?sequence=1&isAllowed=y
#   http://nucleartalent.github.io/Course2ManyBodyMethods/doc/pub/inf/html/inf.html
#   https://www.osti.gov/servlets/purl/1376385
#   http://nucleartalent.github.io/Course2ManyBodyMethods/doc/web/course.html
#   https://publications.nscl.msu.edu/thesis/%20Lietz_2019_5748.pdf
##################################################

##################################################
# OUTLINE
# RS = 0.1
#   rs_0_1_N_74()
#   rs_0_1_N_98
#   rs_0_1_N_114
#   rs_0_1_N_138
#   rs_0_1_N_178
# RS = 0.5
#   rs_0_5_N_10
#   rs_0_5_N_74
#   rs_0_5_N_98
#   rs_0_5_N_114
#   rs_0_5_N_138
#   rs_0_5_N_178
# RS = 1
#   rs_1_N_10
#   rs_1_N_26
#   rs_1_N_42
#   rs_1_N_74
#   rs_1_N_114
#   rs_1_N_122
#   rs_1_N_138
#   rs_1_N_178
# RS = 2
#   rs_2_N_10
#   rs_2_N_74
#   rs_2_N_114
#   rs_2_N_138
#   rs_2_N_178
##################################################

##################################################
# The following applies to all functions in this code:
# 
# Inputs:
#   None.
# Returns:
#   name (a string): a string that describes the data set.  Can be used as a file prefix for saving
#       data
#   training_dim (an int): the recommended size of the training set for machine learning, meaning that
#       the recommended training set is the first dim points from nOrbits and correlation_energy.
#   nOrbits (a numpy array): the number of orbits for each data point.  This is recommended as the x
#       data for analysis.
#   correlation_energy (a numpy array): the couple cluster doubles energy corresponding to each data point.
#       This is recommended as the y data for analysis.
#   N (an int): the number of particles for the data set.
# Stores and returns data corresponding to a couple cluster calculation on an electron gas with the rs and N
# values specified in the function name.
##################################################

##############################
# IMPORTS
##############################
# THIRD-PARTY IMPORTS
# For arrays
import numpy as np

##############################
# RS = 0.1
##############################
# RS_0_1_N_74
def rs_0_1_N_74():
    N = 74
    name = 'Electron_Gas_rs_0_1_N_74'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.19768E+00, -0.24844E+00, -0.27287E+00, -0.28353E+00, -0.29048E+00, -0.29765E+00, -0.30161E+00, -0.30567E+00, -0.30831E+00, -0.31021E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_1_N_98
def rs_0_1_N_98():
    N = 98
    name = 'Electron_Gas_rs_0_1_N_98'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.17885E+00, -0.24386E+00, -0.27757E+00, -0.29325E+00, -0.30352E+00, -0.31946E+00, -0.32511E+00, -0.32873E+00, -0.33133E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_1_N_114
def rs_0_1_N_114():
    N = 114
    name = 'Electron_Gas_rs_0_1_N_114'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.14637E+00, -0.21891E+00, -0.25798E+00, -0.27688E+00, -0.28956E+00, -0.30908E+00, -0.31587E+00, -0.32019E+00, -0.32327E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_1_N_138
def rs_0_1_N_138():
    N = 138
    name = 'Electron_Gas_rs_0_1_N_138'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.10139E+00, -0.18351E+00, -0.22945E+00, -0.25259E+00, -0.26873E+00, -0.28530E+00, -0.29404E+00, -0.30264E+00, -0.30805E+00, -0.31187E+00])			
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_1_N_178
def rs_0_1_N_178():
    N = 178
    name = 'Electron_Gas_rs_0_1_N_178'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.58766E-01, -0.15443E+00, -0.21036E+00, -0.23972E+00, -0.26108E+00, -0.29665E+00, -0.30862E+00, -0.31599E+00, -0.32111E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

##############################
# RS = 0.5
##############################
# RS_0_5_N_10
def rs_0_5_N_10():
    N = 10
    name = 'Electron_Gas_rs_0_5_N_10'
    training_dim = 16
    nOrbits = np.array([26, 50, 74, 98, 242, 394, 570, 914, 1266, 1642, 2010, 2418, 2810, 3210, 3586, 4418, 5202, 6482, 7738, 9946])
    correlation_energy = np.array([-0.13744E+00, -0.19731E+00, -0.21989E+00, -0.22829E+00, -0.24149E+00, -0.24424E+00, -0.24551E+00, -0.24651E+00, -0.24696E+00, -0.24722E+00, -0.24737E+00, -0.24749E+00, -0.24757E+00, -0.24763E+00, -0.24768E+00, -0.24775E+00, -0.24779E+00, -0.24785E+00, -0.24788E+00, -0.24792E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_5_N_74
def rs_0_5_N_74():
    N = 74
    name = 'Electron_Gas_rs_0_5_N_74'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.15357E+00, -0.19885E+00, -0.21935E+00, -0.22749E+00, -0.23250E+00, -0.23746E+00, -0.24012E+00, -0.24278E+00, -0.24447E+00, -0.24567E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_5_N_98
def rs_0_5_N_98():
    N = 98
    name = 'Electron_Gas_rs_0_5_N_98'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.12590E+00, -0.18326E+00, -0.21284E+00, -0.22582E+00, -0.23375E+00, -0.24121E+00, -0.24506E+00, -0.24883E+00, -0.25118E+00, -0.25283E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_5_N_114
def rs_0_5_N_114():
    N = 114
    name = 'Electron_Gas_rs_0_5_N_114'
    training_dim = 10
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3210, 3586, 5642])
    correlation_energy = np.array([-0.10321E+00, -0.16622E+00, -0.20087E+00, -0.21708E+00, -0.22733E+00, -0.23683E+00, -0.24161E+00, -0.24620E+00, -0.24836E+00, -0.24902E+00, -0.25099E+00])	
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_5_N_138
def rs_0_5_N_138():
    N = 138
    name = 'Electron_Gas_rs_0_5_N_138'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.70119E-01, -0.13876E+00, -0.17981E+00, -0.20029E+00, -0.21407E+00, -0.22722E+00, -0.23360E+00, -0.23957E+00, -0.24317E+00, -0.24564E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_0_5_N_178
def rs_0_5_N_178():
    N = 178
    name = 'Electron_Gas_rs_0_5_N_178'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.35714E-01, -0.10863E+00, -0.15715E+00, -0.18338E+00, -0.20227E+00, -0.23158E+00, -0.24021E+00, -0.24523E+00, -0.24859E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

##############################
# RS = 1
##############################
# RS_1_N_10
def rs_1_N_10():
    N = 10
    name = 'Electron_Gas_rs_1_N_10'
    training_dim = 16
    nOrbits = np.array([26, 50, 74, 98, 242, 394, 570, 914, 1266, 1642, 2010, 2418, 2810, 3210, 3586, 4418, 5202, 6482, 7738, 9946])
    correlation_energy = np.array([-0.10969E+00, -0.16221E+00, -0.18024E+00, -0.18623E+00, -0.19486E+00, -0.19654E+00, -0.19730E+00, -0.19789E+00, -0.19815E+00, -0.19830E+00, -0.19839E+00, -0.19846E+00, -0.19850E+00, -0.19854E+00, -0.19856E+00, -0.19860E+00, -0.19863E+00, -0.19866E+00, -0.19868E+00, -0.19870E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_1_N_26
def rs_1_N_26():
    N = 26
    name = 'Electron_Gas_rs_1_N_26'
    training_dim = 16
    nOrbits = np.array([50, 74, 98, 122, 162, 194, 218, 242, 274, 298, 338, 370, 394, 442, 466, 498, 522, 570, 602, 730, 914, 1266, 1642, 2010, 2418, 2810, 3210, 3586, 4002, 4418, 4834, 5202, 5604])
    correlation_energy = np.array([-0.67039E-01, -0.11728E+00, -0.14399E+00, -0.16185E+00, -0.17787E+00, -0.18554E+00, -0.18859E+00, -0.19101E+00, -0.19336E+00, -0.19458E+00, -0.19616E+00, -0.19721E+00, -0.19776E+00, -0.19876E+00, -0.19913E+00, -0.19959E+00, -0.19988E+00, -0.20039E+00, -0.20068E+00, -0.20153E+00, -0.20229E+00, -0.20309E+00, -0.20354E+00, -0.20381E+00, -0.20400E+00, -0.20414E+00, -0.20424E+00, -0.20431E+00, -0.20437E+00, -0.20442E+00, -0.20447E+00, -0.20450E+00, -0.20453E+00])	
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, 26

# RS_1_N_42
def rs_1_N_42():
    N = 42
    name = 'Electron_Gas_rs_1_N_42'
    training_dim = 16
    nOrbits = np.array([50, 74, 98, 122, 162, 194, 218, 242, 274, 298, 338, 370, 394, 442, 466, 498, 522, 570, 730, 914, 1266, 1642, 2010, 2418, 2810, 3210, 3586, 4002, 4418, 4834])
    correlation_energy = np.array([-0.71004E-02, -0.47129E-01, -0.78456E-01, -0.10326E+00, -0.13055E+00, -0.14624E+00, -0.15389E+00, -0.15986E+00, -0.16621E+00, -0.16943E+00, -0.17349E+00, -0.17600E+00, -0.17728E+00, -0.17948E+00, -0.18030E+00, -0.18127E+00, -0.18187E+00, -0.18293E+00, -0.18522E+00, -0.18669E+00, -0.18819E+00, -0.18902E+00, -0.18950E+00, -0.18985E+00, -0.19009E+00, -0.19026E+00, -0.19039E+00, -0.19050E+00, -0.19059E+00, -0.19067E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_1_N_74
def rs_1_N_74():
    N = 74
    name = 'Electron_Gas_rs_1_N_74'
    training_dim = 16
    nOrbits = np.array([162, 194, 218, 242, 274, 298, 338, 370, 394, 570, 730, 914, 1266, 1642, 2010, 2418, 2810, 3210, 3586, 5642])
    correlation_energy = np.array([-0.75782E-01	, -0.98411E-01, -0.11056E+00, -0.12147E+00, -0.13362E+00, -0.14067E+00, -0.15050E+00, -0.15708E+00, -0.16085E+00, -0.17771E+00, -0.18374E+00, -0.18722E+00, -0.19052E+00, -0.19223, -0.19320, -0.19389, -0.19435, -0.19469E+00, -0.19494E+00, -0.19566E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_1_N_114
def rs_1_N_114():
    N = 114
    name = 'Electron_Gas_rs_1_N_114'
    training_dim = 12
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2010, 2418, 2810, 3210, 3586, 5642])
    correlation_energy = np.array([-0.76785E-01, -0.13045E+00, -0.16049E+00, -0.17413E+00, -0.18227E+00, -0.18917E+00, -0.19240E+00, -0.19416E+00, -0.19538E+00, -0.19617E+00, -0.19675E+00, -0.19716E+00, -0.19837E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_1_N_122
def rs_1_N_122():
    N = 122
    name = 'Electron_Gas_rs_1_N_122'
    training_dim = 10
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2010, 2418, 2810, 3210])
    correlation_energy = np.array([-0.67778E-01, -0.12278E+00, -0.15503E+00, -0.17008E+00, -0.17935E+00, -0.18727E+00, -0.19090E+00, -0.19285E+00, -0.19420E+00, -0.19506E+00, -0.19570E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_1_N_138
def rs_1_N_138():
    N = 138
    name= 'Electron_Gas_rs_1_N_138'
    training_dim = 10
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2010, 2418, 2810, 3586, 5642])
    correlation_energy = np.array([-0.51117E-01, -0.10783E+00, -0.14360E+00, -0.16132E+00, -0.17285E+00, -0.18305E+00, -0.18754E+00, -0.18991E+00, -0.19152E+00, -0.19254E+00, -0.19382E+00, -0.19536E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_1_N_178
def rs_1_N_178():
    N = 178
    name = 'Electron_Gas_rs_1_N_178'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.24362E-01, -0.80981E-01, -0.12228E+00, -0.14516E+00, -0.16151E+00, -0.18524E+00, -0.19125E+00, -0.19454E+00, -0.19666E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

##############################
# RS = 2
##############################
# RS_2_N_10
def rs_2_N_10():
    N = 10
    name = 'Electron_Gas_rs_2_N_10'
    training_dim = 16
    nOrbits = np.array([26, 50, 74, 98, 242, 394, 570, 914, 1266, 1642, 2010, 2418, 2810, 3210, 3586, 4418, 5202, 6482, 7738, 9946])
    correlation_energy = np.array([-0.78014E-01, -0.11947E+00, -0.13232E+00, -0.13579E+00, -0.14014E+00, -0.14090E+00, -0.14123E+00, -0.14148E+00, -0.14159E+00, -0.14165E+00, -0.14169E+00, -0.14172E+00, -0.14174E+00, -0.14175E+00, -0.14176E+00, -0.14178E+00, -0.14179E+00, -0.14180E+00, -0.14181E+00, -0.14182E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_2_N_74
def rs_2_N_74():
    N = 74
    name = 'Electron_Gas_rs_2_N_74'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.86475E-01, -0.11747E+00, -0.12985E+00, -0.13359E+00, -0.13551E+00, -0.13719E+00, -0.13802E+00, -0.13880E+00, -0.13928E+00, -0.13960E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_2_N_74
def rs_2_N_114():
    N = 74
    name = 'Electron_Gas_rs_2_N_114'
    training_dim = 6
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418])
    correlation_energy = np.array([-0.51317E-01, -0.92486E-01, -0.11608E+00, -0.12645E+00, -0.13213E+00, -0.13621E+00, -0.13782E+00, -0.13909E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_2_N_138
def rs_2_N_138():
    N = 138
    name = 'Electron_Gas_rs_2_N_138'
    training_dim = 8
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642, 2418, 3586, 5642])
    correlation_energy = np.array([-0.33331E-01, -0.75374E-01, -0.10356E+00, -0.11746E+00, -0.12613E+00, -0.13298E+00, -0.13552E+00, -0.13757E+00, -0.13868E+00, -0.13939E+00])
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

# RS_2_N_178
def rs_2_N_178():
    N = 178
    name = 'Electron_Gas_rs_2_N_178'
    training_dim = 6
    nOrbits = np.array([242, 394, 570, 730, 914, 1266, 1642])
    correlation_energy = np.array([-0.14987E-01, -0.54324E-01, -0.85988E-01, -0.10406E+00, -0.11690E+00, -0.12907E+00, -0.13394E+00])			
    assert len(nOrbits) == len(correlation_energy)
    return name, training_dim, nOrbits, correlation_energy, N

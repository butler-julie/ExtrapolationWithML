# Regression Codes

Will be completed later to explain all of the files that relate to the regression extrapolation algorithms


### [Regression Test Code](RegressionTestCode.py)

This is an all inclusive code to test extrapolating with Linear Regression, Ridge Regression, and Kernel Ridge Regression using both regular format training and sequential format training.  This code also includes options for graphing and saving the results.  For a given data set the code will perform hyperparameter tuning on whatever combination of linear regression, ridge regression, kernel ridge regression, regular data formatting, and sequential data formatting that are specified by setting the values of booleans.  The code will print the optimized parameters once found and the best MSE for the extrapolation to the terminal if the verbose boolean is set to True.  Graphing capabilities and saving parameters and extrapolated data sets are also controlled with booleans.

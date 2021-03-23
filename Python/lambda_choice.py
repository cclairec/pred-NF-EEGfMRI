# -*- coding: utf-8 -*-
"""
Estimation of the regularisation parameter : lambda 

Created on Tue Mar 23 16:02:14 2021

@author: caroline-pinte
"""

import numpy as np

def lambda_choice(Dtrain, rep_train, nb_freq_band, reg_function, lambdas, disp_) :
    '''
    Estimate the regularisation parameter : lambda.

            Parameters:
                    Dtrain (array): design matrix for learning.
                    rep_train (array): NF variable for learning.
                    nb_freq_band (int): number of freq bands (default 10).
                    reg_function (string): regularisation function, must be 'lasso' (matlab), 'fistaL1' or 'L12'.
                    lambdas (array): list of int values to be tested.
                    disp_ (bool): display figure or not.
                    
            Returns:
                    regul_lambda (int): regularisation parameter lambda.
    '''
    
    K = 10
    
    if (rep_train.ndim == 2) :
        if (np.shape(rep_train)[0] == 2) or (np.shape(rep_train)[1] == 2) :
            weights = [1/3, 1/2, 2/3]
            if (np.shape(rep_train)[0] == 2) :
                NF_1 = rep_train[0,:]
                NF_2 = rep_train[1,:]
            elif (np.shape(rep_train)[1] == 2) :
                NF_1 = rep_train[:,0]
                NF_2 = rep_train[:,1]                
    # elif (np.shape(rep_train)[0] == 2) :
    #     weights = [1/3, 1/2, 2/3]
    #     NF_1 = rep_train[0,:]
    #     NF_2 = rep_train[1,:]
    else :
        weights = -1
    
    sparsity_threshold = np.shape(Dtrain)[1] * 40/100 # we want at most 40 % of nonzeros coefficients
# delay_cv = (size(Dtrain,1)-size_cv_dataset)/K;
    size_cv_dataset = np.ceil(np.shape(Dtrain)[0]/10)
    delay_cv = (np.shape(Dtrain)[0] - size_cv_dataset) / K
    
    
    
    
    
    
    
    
    
    
    
    
    return 0
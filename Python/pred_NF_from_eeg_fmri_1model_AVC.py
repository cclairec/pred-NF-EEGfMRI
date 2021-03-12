# -*- coding: utf-8 -*-
"""
Model estimation and NF predictions

Created on Thu Mar 11 15:13:15 2021

@author: caroline-pinte
"""

import numpy as np
import logging
from colorlog import ColoredFormatter
import mat73

#====================================================================
# Initialisation : Format of logs
#====================================================================

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(asctime)s%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
#LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(
	LOGFORMAT,
	datefmt='%d/%m/%Y %H:%M:%S ',
	reset=True,
	log_colors={
		'DEBUG':    'bold_cyan',
		'INFO':     'bold_white',
		'WARNING':  'bold_yellow',
		'ERROR':    'bold_red',
		'CRITICAL': 'bold_red,bg_white',
	},
	secondary_log_colors={},
	style='%'
)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('pythonConfig')
logger.setLevel(LOG_LEVEL)

if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(stream)

#====================================================================
# Main function
#====================================================================

def pred_NF_from_eeg_fmri_1model_AVC(dataPath, suj_ID, session, learn_run, test_run, mod='fmri', nb_bandfreq=10, reg_function='fistaL1', clean_test=1) :
    '''
    Estimate model and predict NF scores.

            Parameters:
                    dataPath (string): path to retrieve data.
                    suj_ID (string): the patient ID.
                    session (string): the session, must contain subfolders MI_PRE, NF1, NF2, NF3.
                    learn_run (string): the run that we will use for learning, must be NF1 NF2 or NF3.
                    test_run (string): the run that we will use for testing, must be NF1 NF2 or NF3.
                    mod (string): model to learn, must be 'eeg', 'fmri' or 'both'. Here 'both' means 2 models.
                    nb_bandfreq (int): number of freq bands (default 10).
                    reg_function (string): regularisation function, must be 'lasso' (matlab), 'fistaL1' or 'L12'.
                    clean_test (bool): clean or not the design matrix of data test.

            Returns:
                    Res (dictionary): contains result model and values in a format that will fit Matlab.
    '''
    
    logger.info("Beginning : {} {} Learn{} Test{}".format(suj_ID,session,learn_run,test_run))
    
    ### Loading data
    
    suj_learning_EEG = mat73.loadmat("{}/{}/{}_NFEEG/{}/{}/EEG_features_Laplacian.mat".format(dataPath,suj_ID,suj_ID,session,learn_run))

    suj_learning_fMRI = mat73.loadmat("{}/{}/{}_NFfMRI/{}/roi_all_sessions/{}/fMRI_features_sma_and_m1.mat".format(dataPath,suj_ID,suj_ID,session,learn_run))
  
    suj_testing_EEG = mat73.loadmat("{}/{}/{}_NFEEG/{}/{}/EEG_features_Laplacian.mat".format(dataPath,suj_ID,suj_ID,session,test_run))
   
    suj_testing_fMRI = mat73.loadmat("{}/{}/{}_NFfMRI/{}/roi_all_sessions/{}/fMRI_features_sma_and_m1.mat".format(dataPath,suj_ID,suj_ID,session,test_run))
    
    ### Reshaping EEG signals
    
    ### Removing bad segments
    
    ### Removing the corresponding removed times to the NF scores
    
    ### Load Channel names
    
    ### Extracting NF_EEG / NF_fMRI scores
    
    ### Compute the design matrices for learning and test
    
    ### Removing some eletrodes
    
    ### Estimate design matrix for fMRI model
    
    ### Execution : estimation regul param lambda
    
    ### Optimization : Forward backward optimization
    
    ### Saving results into a Matlab object
    a = np.arange(20)
    Res = {"a": a, "label": "experiment"}
    return Res
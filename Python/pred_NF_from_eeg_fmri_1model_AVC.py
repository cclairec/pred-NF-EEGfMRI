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
#import scipy.io as sio
import math
import scipy 
from scipy import stats
from scipy.signal import savgol_filter

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
# Initialisation : Format of logs
#====================================================================

def psdbandpower(Pxx, f, fmin, fmax):
    colPxx = Pxx
    colW = f
    
    idx = np.argwhere(colW<=fmin)
    idx1 = idx[-1][0]
    idx = np.argwhere(colW>=fmax)
    idx2 = idx[0][0]
    
    W_diff = np.diff(colW)
    lastRectWidth = 0
    width = W_diff.copy()
    width = np.append(width,lastRectWidth)
 
    pwr = np.dot(width[idx1:idx2+1] , colPxx[idx1:idx2+1])
    
    return pwr
    

def bandpower(x, fs, fmin, fmax):
    n = np.shape(x)[0]
    f, Pxx = scipy.signal.periodogram(x, fs=fs, window=scipy.signal.windows.hamming(n), detrend=False)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    res = psdbandpower(Pxx,f,fmin,fmax)
    #res = scipy.integrate.simps(Pxx[ind_min: ind_max], f[ind_min: ind_max])
    return res

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
    
    logger.info("* Beginning : {} {} Learn{} Test{}".format(suj_ID,session,learn_run,test_run))
    
    ### Loading data
    logger.info("* Loading EEG and fMRI data")
    
    suj_learning_EEG = mat73.loadmat("{}/{}/{}_NFEEG/{}/{}/EEG_features_Laplacian.mat".format(dataPath,suj_ID,suj_ID,session,learn_run))
    suj_learning_EEG_FEAT = suj_learning_EEG['EEG_FEAT'] 
    
    suj_learning_fMRI = mat73.loadmat("{}/{}/{}_NFfMRI/{}/roi_all_sessions/{}/fMRI_features_sma_and_m1.mat".format(dataPath,suj_ID,suj_ID,session,learn_run))
    suj_learning_fMRI_FEAT = suj_learning_fMRI['fMRI_FEAT']['feat']
      
    suj_testing_EEG = mat73.loadmat("{}/{}/{}_NFEEG/{}/{}/EEG_features_Laplacian.mat".format(dataPath,suj_ID,suj_ID,session,test_run))
    suj_testing_EEG_FEAT = suj_testing_EEG['EEG_FEAT'] 
        
    suj_testing_fMRI = mat73.loadmat("{}/{}/{}_NFfMRI/{}/roi_all_sessions/{}/fMRI_features_sma_and_m1.mat".format(dataPath,suj_ID,suj_ID,session,test_run))
    suj_testing_fMRI_FEAT = suj_testing_fMRI['fMRI_FEAT']['feat']

    # Extra parameters
    disp_fig = 1
    smooth_param = 7 # window size >3 and odd
    nbloc = 8
    blocsize = 160
    f_m = 7 # minimum freq to consider
    f_M = 30 # maximum freq to consider
    f_win = math.ceil((f_M-f_m)/nb_bandfreq) # windows frequency size
    
    ### Reshaping EEG signals
    logger.info("* Reshaping EEG signals of learning and testing sessions")
    
    EEG_signal_reshape_learning = np.reshape(suj_learning_EEG_FEAT['data'], (64,64000), order="F")
    EEG_signal_reshape_test = np.reshape(suj_testing_EEG_FEAT['data'], (64,64000), order="F")
    
    ### Removing bad segments
    logger.info("* Removing bad segments from EEG signals")
    logger.info("Bad segments already removed")
    
    ### Identifying the corresponding removed times to the NF scores
    logger.info("* Identifying the corresponding removed times")
    
    badseg_learning = suj_learning_EEG_FEAT['bad_segments'][::50].copy()
    bad_scores_learning_ind = np.nonzero(badseg_learning)[0]
    
    badseg_testing = suj_testing_EEG_FEAT['bad_segments'][::50].copy()
    bad_scores_testing_ind = np.nonzero(badseg_testing)[0]

    ### Load Channel names -> not used
    #logger.info("* Loading channel names")
    #chanlocs = sio.loadmat("{}/Chanlocs.mat".format(dataPath))
    
    ### Extracting NF_EEG / NF_fMRI scores
    logger.info("* Extracting NF_EEG scores")

    X_eeg_learn_smooth_Lap = stats.zscore(suj_learning_EEG_FEAT['smoothnf'], axis=0, ddof=1)
    X_eeg_test_smooth_Lap = stats.zscore(suj_testing_EEG_FEAT['smoothnf'], axis=0, ddof=1)
    
    logger.info("* Extracting NF_fMRI scores")
    
    fmri_NF_learn = suj_learning_fMRI_FEAT[0]['nf']
    fmri_NF_test = suj_testing_fMRI_FEAT[0]['nf']
    
    kk = -1
    fmri_NF_reshape = np.zeros(1280)
    fmri_NF_reshape_test = np.zeros(1280)
    
    for ii in range(0,len(X_eeg_learn_smooth_Lap),4) :
        kk = kk+1
        for k in range(ii, ii+4) :
            fmri_NF_reshape[k] = fmri_NF_learn[kk]
            fmri_NF_reshape_test[k] = fmri_NF_test[kk]
    
    X_fmri_reshape = stats.zscore(fmri_NF_reshape, axis=0, ddof=1);
    X_fmri_reshape_test = stats.zscore(fmri_NF_reshape_test, axis=0, ddof=1);

    ### Smooth NF scores
    logger.info("* Smoothing NF scores")
    
    X_fmri_reshape_learn_smooth = savgol_filter(X_fmri_reshape, smooth_param, 3)
    X_fmri_reshape_test = savgol_filter(X_fmri_reshape_test, smooth_param, 3)

    ### Removing bad segments
    logger.info("* Removing bad segments from NF scores")
    
    X_fmri_reshape_learn_smooth = X_fmri_reshape_learn_smooth * 50
    X_fmri_reshape_learn_smooth = np.delete(X_fmri_reshape_learn_smooth, bad_scores_learning_ind)
    
    X_fmri_reshape_test = X_fmri_reshape_test * 50
    X_fmri_reshape_test = np.delete(X_fmri_reshape_test, bad_scores_testing_ind)
    
    X_eeg_learn_smooth_Lap = X_eeg_learn_smooth_Lap * 50
    X_eeg_learn_smooth_Lap = np.delete(X_eeg_learn_smooth_Lap, bad_scores_learning_ind)
    
    X_eeg_test_smooth_Lap = X_eeg_test_smooth_Lap * 50
    X_eeg_test_smooth_Lap = np.delete(X_eeg_test_smooth_Lap, bad_scores_testing_ind)
    
    logger.info("* NF score to be learned, with mod : {}".format(mod))

    if (mod == 'eeg') :
        logger.info("Mod chosen : {}".format(mod))
        #X = X_eeg_learn_smooth_Lap
        #X_test = X_eeg_test_smooth_Lap
        logger.info("Not implemented")
    elif (mod == 'fmri') :
        logger.info("Mod chosen : {}".format(mod))
        X = X_fmri_reshape_learn_smooth
        X_test = X_fmri_reshape_test
    elif (mod == 'both') :
        logger.info("Mod chosen : {}".format(mod))
        #X = [X_eeg_learn_smooth_Lap + X_fmri_reshape_learn_smooth]
        #X_test = [X_eeg_test_smooth_Lap+ X_fmri_reshape_test]
        #weight = 0.5
        logger.info("Not implemented")
            
    ### Compute the design matrices for learning and test
    logger.info("* Computing the design matrices")
    
    # Initialisation
    freq_band_learning = []
    freq_band_test = []
    f_interval = []
    for ff in range(0,nb_bandfreq) :
        freq_band_learning.append(np.zeros((1281,np.shape(EEG_signal_reshape_learning)[0])))
        freq_band_test.append(np.zeros((1281,np.shape(EEG_signal_reshape_learning)[0])))
        f_interval.append([0,0])
    f_interval.append([0,0]) # one more
    
# k=1; clear freq_band_*;
# steps = 400; 
# for i=1:50:64000, % shift for 1/4 of second
#     k=k+1;
#     f_interval{1} = [f_m f_m+f_win]; 
#     for ff=1:nb_bandfreq
# def bandpower(x, fs, fmin, fmax):
# bandpower(x,fs,freqrange)
#         freq_band_learning{ff}(k,:)=(bandpower(EEG_signal_reshape_learning(:,i:min(size(EEG_signal_reshape_learning,2),i+steps))',200,f_interval{ff}));
#         freq_band_test{ff}(k,:)=(bandpower(EEG_signal_reshape_test(:,i:min(size(EEG_signal_reshape_test,2),i+steps))',200,f_interval{ff}));
#         f_interval{ff+1} = [(max(f_interval{ff})-1) (max(f_interval{ff})-1) + f_win];
#     end
# end
    k=0
    steps = 400
    
    for i in range(0,64000,50) :
        k = k+1
        if (k%100 == 0) :
            logger.info("Computing ... {}/1280".format(k))
        elif (k==1280) :
            logger.info("Done : {}/1280".format(k))
        f_interval[0] = [f_m,f_m+f_win]
        
        for ff in range(0,nb_bandfreq) :
            eeg_signal_learn = EEG_signal_reshape_learning[:,i:min((np.shape(EEG_signal_reshape_learning)[1]),i+steps+1)].T
            
            for col_index in range(0,np.shape(eeg_signal_learn)[1]) :
                col = eeg_signal_learn[:,col_index]
                freq_band_learning[ff][k,col_index] = bandpower(col, 200, f_interval[ff][0], f_interval[ff][1])
            
            eeg_signal_test = EEG_signal_reshape_test[:,i:min((np.shape(EEG_signal_reshape_test)[1]),i+steps+1)].T
            
            for col_index in range(0,np.shape(eeg_signal_test)[1]) :
                col = eeg_signal_test[:,col_index]
                freq_band_test[ff][k,col_index] = bandpower(col, 200, f_interval[ff][0], f_interval[ff][1])
            
            f_interval[ff+1] = [(max(f_interval[ff])-1),(max(f_interval[ff])-1)+f_win]  
        

    ### Removing some eletrodes
    
    ### Estimate design matrix for fMRI model
    
    ### Execution : estimation regul param lambda
    
    ### Optimization : Forward backward optimization
    print('debug')
    
    ### Saving results into a Matlab-like object
    logger.info("* Saving results")

    learn_run = 0
    test_run = 0
    alpha = 0
    NF_estimated = 0
    filter_estimated_eeg = 0
    filter_estimated_fmri = 0
    
    X_fmri_reshape_test = 0
    X_eeg_test_smooth_Lap = 0
    X_eeg_learn_smooth_Lap = 0
    X_fmri_reshape_learn_smooth = 0
    rep_test = 0
    
    nb_bandfreq = 0
    regul_eeg = 0
    t = 0
    D_test = 0
    D_learning = 0
    index_freq_band_used = 0
    f_interval = 0
    elect_kept = 0
    bad_scores_testing_ind = 0
    bad_scores_learning_ind = 0

    Res = {'learning_session':learn_run, \
           'test_session':test_run, \
           'alpha':alpha, \
           'NF_estimated_fMRI':NF_estimated, \
           'filter_estimated_e':filter_estimated_eeg, \
           'filter_estimated_f':filter_estimated_fmri, \
           'NF_fMRI_test':X_fmri_reshape_test, \
           'NF_EEG_test':X_eeg_test_smooth_Lap, \
           'NF_EEG_learn':X_eeg_learn_smooth_Lap, \
           'NF_fMRI_learn':X_fmri_reshape_learn_smooth, \
           'rep_test':rep_test, \
           'nb_bandfreq':nb_bandfreq, \
           'lambda_value':regul_eeg, \
           'time':t, \
           'D_test':D_test, \
           'D_learn':D_learning, \
           'index_freq_band_used':index_freq_band_used, \
           'f_interval':f_interval, \
           'elect_used':elect_kept, \
           'bad_scores_testing_ind':bad_scores_testing_ind, \
           'bad_scores_learning_ind':bad_scores_learning_ind \
          }
    
    logger.info("* Done !")
    return Res
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
import pickle
import time
from matplotlib import pyplot as plt
from lambda_choice import lambda_choice
from forward_backward_optimisation import forward_backward_optimisation


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
# Utilities
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

def spm_hrf(RT,p):
    p = [float(x) for x in p]
    fMRI_T = 16.0
    RT = float(RT)
    dt = RT/fMRI_T
    u = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf = scipy.stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    index = np.arange(0,(p[6]/RT)+1,dtype=int) * int(fMRI_T)
    hrf = hrf[index]
    hrf = hrf/np.sum(hrf)
    return hrf

#====================================================================
# Main function
#====================================================================

def pred_NF_from_eeg_fmri_1model_AVC(dataPath, resPath, suj_ID, session, learn_run, test_run, mod='fmri', nb_bandfreq=10, reg_function='fistaL1', clean_test=1, electrodes='all') :
    '''
    Estimate model and predict NF scores.

            Parameters:
                    dataPath (string): path to retrieve data.
                    resPath (string): path to save results.
                    suj_ID (string): the patient ID.
                    session (string): the session, must contain subfolders MI_PRE, NF1, NF2, NF3.
                    learn_run (string): the run that we will use for learning, must be NF1 NF2 or NF3.
                    test_run (string): the run that we will use for testing, must be NF1 NF2 or NF3.
                    mod (string): model to learn, must be 'eeg', 'fmri' or 'both'. Here 'both' means 2 models.
                    nb_bandfreq (int): number of freq bands (default 10).
                    reg_function (string): regularisation function, must be 'lasso' (matlab), 'fistaL1' or 'L12'.
                    clean_test (bool): clean or not the design matrix of data test.
                    electrodes (string): use all channels or only motor channels, must be 'all' or 'motor'
                    
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
    logger.info("--- Bad segments already removed")
    
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
        logger.info("--- Mod chosen : {}".format(mod))
        #X = X_eeg_learn_smooth_Lap
        #X_test = X_eeg_test_smooth_Lap
        logger.info("Not implemented")
    elif (mod == 'fmri') :
        logger.info("--- Mod chosen : {}".format(mod))
        X = X_fmri_reshape_learn_smooth
        X_test = X_fmri_reshape_test
    elif (mod == 'both') :
        logger.info("--- Mod chosen : {}".format(mod))
        #X = [X_eeg_learn_smooth_Lap + X_fmri_reshape_learn_smooth]
        #X_test = [X_eeg_test_smooth_Lap+ X_fmri_reshape_test]
        #weight = 0.5
        logger.info("Not implemented")
    else :
        logger.error("mod (string): model to learn, must be 'eeg', 'fmri' or 'both'.")
        return 0
            
    ### Removing some eletrodes
    logger.info("* Removing noisy electrodes")
    
    motor_channels = [4,5,17,20,21,22,23,24,25,26,27,34,35,40,41,42,43,48,49,63] 
    frontal_channels = [32,33,16] # electrodes to keep, base 64 already. Removed for patients.
    all_channels = np.arange(0,64)
    ind_elect = np.arange(0,64) # electrodes to exclude
    if (electrodes == 'all') :
        logger.info("--- All electrodes chosen")
        ind_elect_eeg_exclud = []
    elif (electrodes == 'motor') :
        logger.info("--- Motor electrodes chosen")
        ind_elect_eeg_exclud = [element for element in ind_elect if element not in motor_channels] 
    else :
        logger.error("electrodes (string): use all channels or only motor channels, must be 'all' or 'motor'")
        return 0
    
    ### Compute the design matrices for learning and testing
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
    
    # Compute freq_band for learning and testing
    
    ################ debug ##################
    
    # k=0
    # steps = 400
    
    # for i in range(0,64000,50) :
    #     k = k+1
    #     if (k%100 == 0) :
    #         logger.info("Computing ... {}/1280 rows".format(k))
    #     elif (k==1280) :
    #         logger.info("Done : {}/1280 rows".format(k))
    #     f_interval[0] = [f_m,f_m+f_win]
        
    #     for ff in range(0,nb_bandfreq) :
    #         eeg_signal_learn = EEG_signal_reshape_learning[:,i:min((np.shape(EEG_signal_reshape_learning)[1]),i+steps+1)].T
            
    #         for col_index in range(0,np.shape(eeg_signal_learn)[1]) :
    #             col = eeg_signal_learn[:,col_index]
    #             freq_band_learning[ff][k,col_index] = bandpower(col, 200, f_interval[ff][0], f_interval[ff][1])
            
    #         eeg_signal_test = EEG_signal_reshape_test[:,i:min((np.shape(EEG_signal_reshape_test)[1]),i+steps+1)].T
            
    #         for col_index in range(0,np.shape(eeg_signal_test)[1]) :
    #             col = eeg_signal_test[:,col_index]
    #             freq_band_test[ff][k,col_index] = bandpower(col, 200, f_interval[ff][0], f_interval[ff][1])
            
    #         f_interval[ff+1] = [(max(f_interval[ff])-1),(max(f_interval[ff])-1)+f_win]  
        

    
    # with open("freq_band_learning.txt", "wb") as fp:   #Pickling
    #     pickle.dump(freq_band_learning, fp)
        
    # with open("freq_band_test.txt", "wb") as fp:   #Pickling
    #     pickle.dump(freq_band_test, fp)
    
    with open("freq_band_learning.txt", "rb") as fp:   # Unpickling
        freq_band_learning = pickle.load(fp)

    with open("freq_band_test.txt", "rb") as fp:   # Unpickling
        freq_band_test = pickle.load(fp)
    
    ###################################""
    
    # D_learning
    index_freq_band_used = np.arange(0,len(freq_band_learning))
    D_learning = np.hstack(freq_band_learning)
    sz = np.shape(D_learning)[0]
    D_learning = D_learning[1:(sz+1),:]
    sz = np.shape(D_learning)[0]
    D_learning = np.reshape(D_learning, (sz,64,nb_bandfreq), order="F")
    D_learning[:,ind_elect_eeg_exclud,:] = 0
    
    # D_test
    D_test = np.hstack(freq_band_test)
    sz = np.shape(D_test)[0]
    D_test = D_test[1:(sz+1),:]
    sz = np.shape(D_test)[0]
    D_test = np.reshape(D_test, (sz,64,nb_bandfreq), order="F")
    D_test[:,ind_elect_eeg_exclud,:] = 0
      
    # Estimate design matrix for fMRI model
    p = [4,16,1,1,3,0,32]
    hrf4 = spm_hrf( 1/4, p ) # 4Hz
    
    p = [5,16,1,1,3,0,32]
    hrf5 = spm_hrf( 1/4, p )
    
    p = [3,16,1,1,3,0,32]
    hrf3 = spm_hrf( 1/4, p )
    
    D_learning_ = np.zeros((np.shape(D_learning)[0],np.shape(D_learning)[1] * 3,np.shape(D_learning)[2]))
    D_test_ = np.zeros((np.shape(D_test)[0],np.shape(D_test)[1] * 3,np.shape(D_test)[2]))
    
    for i in range(0, np.shape(D_learning)[1]) :
        for j in range(0, np.shape(D_learning)[2]) :

            resp3 = np.convolve(D_learning[:,i,j], hrf3)
            resp3 = resp3[0:np.shape(D_learning[:,i,j])[0]]
            D_learning_[:,i,j] = resp3
            
            resp3 = np.convolve(D_test[:,i,j], hrf3)
            resp3 = resp3[0:np.shape(D_test[:,i,j])[0]]
            D_test_[:,i,j] = resp3
            
            resp4 = np.convolve(D_learning[:,i,j], hrf4)
            resp4 = resp4[0:np.shape(D_learning[:,i,j])[0]]
            D_learning_[:,i+np.shape(D_learning)[1],j] = resp4
        
            resp4 = np.convolve(D_test[:,i,j], hrf4)
            resp4 = resp4[0:np.shape(D_test[:,i,j])[0]]
            D_test_[:,i+np.shape(D_learning)[1],j] = resp4
        
            resp5 = np.convolve(D_learning[:,i,j], hrf5)
            resp5 = resp5[0:np.shape(D_learning[:,i,j])[0]]
            D_learning_[:,i+2*np.shape(D_learning)[1],j] = resp5
        
            resp5 = np.convolve(D_test[:,i,j], hrf5)
            resp5 = resp5[0:np.shape(D_test[:,i,j])[0]]
            D_test_[:,i+2*np.shape(D_learning)[1],j] = resp5

    D_learning = np.delete(D_learning, bad_scores_learning_ind, 0)
    D_test = np.delete(D_test, bad_scores_testing_ind, 0)
    D_learning_ = np.delete(D_learning_, bad_scores_learning_ind, 0)
    D_test_ = np.delete(D_test_, bad_scores_testing_ind, 0)
    
    ### Cleaning the design matrices for learning and testing
    logger.info("* Cleaning the design matrices")
    
    # Cleaning the design matrix for learning step, by thresholding bad observations
    mean_3std_learn = np.mean(np.mean(D_learning[(D_learning!=0)])) + 3*np.mean(np.std(D_learning[(D_learning!=0)]))
    D_learning[(D_learning>mean_3std_learn)] = mean_3std_learn
    
    mean_3std_learn = np.mean(np.mean(D_learning_[(D_learning_!=0)])) + 3*np.mean(np.std(D_learning_[(D_learning_!=0)]))
    D_learning_[(D_learning_>mean_3std_learn)] = mean_3std_learn
    
    # Cleaning the design matrix for test step, by thresholding bad observations
    if (clean_test==1) :

        mean_3std_test = np.mean(np.mean(D_test[(D_test!=0)])) + 3*np.mean(np.std(D_test[(D_test!=0)]))
        D_test[(D_test>mean_3std_test)] = mean_3std_test
    
        mean_3std_test = np.mean(np.mean(D_test_[(D_test_!=0)])) + 3*np.mean(np.std(D_test_[(D_test_!=0)]))
        D_test_[(D_test_>mean_3std_test)] = mean_3std_test
    
    # Prepare D_learning_old and D_test_old
    if (mod == 'both') or (mod == 'fmri') :
        D_learning_old = np.zeros((np.shape(D_learning)[0],np.shape(D_learning)[1] + np.shape(D_learning_)[1],np.shape(D_learning)[2]))
        D_test_old = np.zeros((np.shape(D_test)[0],np.shape(D_test)[1] + np.shape(D_test_)[1],np.shape(D_test)[2]))
        
        for i in range(0, np.shape(D_learning)[0]) :
            D_learning_old[i,:,:] = np.vstack( [np.squeeze(D_learning[i,:,:]), np.squeeze(D_learning_[i,:,:])] )

        for i in range(0, np.shape(D_test)[0]) :
            D_test_old[i,:,:] = np.vstack( [np.squeeze(D_test[i,:,:]), np.squeeze(D_test_[i,:,:])] )

    elif (mod == 'eeg') :
        D_learning_old = D_learning.copy()
        D_test_old = D_test.copy()
        
    else :
        logger.error("mod (string): model to learn, must be 'eeg', 'fmri' or 'both'.")
        return 0
    
    ### Execution 
    logger.info("* Execution ...")
    
    # Cases of same run used
    if (learn_run == test_run) :
        learning_block = np.arange( blocsize*1 , np.round(np.shape(D_learning_old)[0]/2) , dtype=int )
        testing_block = np.arange( learning_block[-1]+1 , np.shape(D_test_old)[0] , dtype=int )
    else :
        learning_block = np.arange( blocsize*1 , np.shape(D_learning_old)[0] , dtype=int )
        testing_block = np.arange( 0 , np.shape(D_test_old)[0] , dtype=int )
    
    # Timer
    tic = time.time()
    
    testing_dummy_data = 0
    if (testing_dummy_data == 1) :
        logger.error("Not implemented")
        return 0
    else :
        rep_learning = X[learning_block] # removing the first bloc from learning phase
        rep_test = X_test[testing_block]
        
        D_learning = D_learning_old[learning_block,:,:]
        matrix_ones = np.ones((np.shape(D_learning)[0], np.shape(D_learning)[1]))
        matrix_ones = matrix_ones[..., np.newaxis]
        D_learning = np.concatenate((D_learning,matrix_ones),axis=2)
    
        D_test = D_test_old[testing_block,:,:]
        matrix_ones = np.ones((np.shape(D_test)[0], np.shape(D_test)[1]))
        matrix_ones = matrix_ones[..., np.newaxis]
        D_test = np.concatenate((D_test,matrix_ones),axis=2)
    
    ### Estimating regularisation parameter lambda 
    logger.info("* Estimating lambda for method : {}".format(reg_function))
    
    if (reg_function == 'lasso') :
        #lambdas=[0.1:0.2:10];
        logger.error("Not implemented")
        return 0
    elif (reg_function == 'fistaL1') :
        lambdas = np.arange(0,2000+1, 100) # initial values
        #lambdas = np.arange(0,50000, 500) # test
    else :
        logger.error("reg_function (string): regularisation function, must be 'lasso' (matlab), 'fistaL1' or 'L12'")
        return 0
    
    regul_eeg = lambda_choice(D_learning, rep_learning, nb_bandfreq, reg_function, lambdas, disp_fig, logger)
    #plt.savefig('{}/Fig1.png'.format(resPath))
    
    plt.gcf().savefig('{}/Fig1.png'.format(resPath))
    plt.show()
    
    logger.info("--- lambda = {}".format(regul_eeg))
    
    ### Forward Backward Optimization with the chosen lambda + estimate NF scores
    logger.info("* Computing optimization with the chosen lambda")
    if (reg_function == 'lasso') :
        logger.error("Not implemented")
        return 0
    
    elif (reg_function == 'fistaL1') :
        alpha = forward_backward_optimisation(D_learning, rep_learning.T, regul_eeg)
        
        NF_estimated = np.zeros(np.shape(D_test)[0])
        for t in range(0,np.shape(D_test)[0]) :
            NF_estimated[t] = np.trace( np.matmul(np.squeeze(D_test[t,:,:]).T,alpha) )

        predicted_values = np.zeros(np.shape(D_learning)[0])
        for t in range(0,np.shape(D_learning)[0]) :
            predicted_values[t] = np.trace( np.matmul(np.squeeze(D_learning[t,:,:]).T,alpha) )
    
    else :
        logger.error("reg_function (string): regularisation function, must be 'lasso' (matlab), 'fistaL1' or 'L12'")
        return 0
    
    # End Timer
    toc = time.time()
    t = toc - tic
    
    ### Smoothing NF scores
    logger.info("* Smoothing the NF scores obtained")
    smooth_NF = 1
    if (smooth_NF == 1) :
        smooth_wind_test = 2
        NF_estimated_notsmoothed = NF_estimated.copy()
        for i in range(0,len(NF_estimated)) :
            if (i < smooth_wind_test) :
                NF_estimated[i] = np.mean(NF_estimated_notsmoothed[0:i+1])
            else :
                NF_estimated[i] = np.mean(NF_estimated_notsmoothed[i-smooth_wind_test:i+1])

    if (mod == 'eeg') :
        weight = 0
        
    ### Preparing results figures
    logger.info("* Preparing results figures")
    if (reg_function == 'fistaL1') :
        # EEG start at 65+64+1=130, since its design matrix is at the end : fMRI4s, fMRI5s, EEG.
        nb_mat_design = int(np.shape(alpha)[0] / 64)
        ch1 = 0
        ch64 = 63
        filter_estimated = []
        for nmd in range(0,nb_mat_design) :
            end = np.shape(alpha)[1]
            alpha_1 = alpha[ch1:ch64+1,0:end-1]
            alpha_1 = np.delete(alpha_1,31,0)
            filter_estimated.append(alpha_1)
            ch1 = ch64 + 1
            ch64 = ch1 + 63

        filter_estimated_eeg = filter_estimated[nb_mat_design-1]
        filter_estimated_fmri = filter_estimated[0:nb_mat_design-1]

    elect_kept = np.ones(64)
    elect_kept[ind_elect_eeg_exclud] = 0
    
    if (disp_fig == 1) :      
        # # plot 1
        # plt.plot(lambdas, CV_mean_, label = "cv error", color='blue')  
        # # plot 2
        # plt.plot(lambdas, Cost_train_mean, label = "training error", color='red')
        # # plot 3
        # plt.plot(lambdas, biais_var, label = "cv error + training error", color='yellow', marker='.')
          
        # plt.xlabel('lambda')
        # plt.ylabel('error')
        # plt.title('lambda choice')
        # plt.legend()
        # # plt.show() # out of this function to save the fig
        logger.info("todo figures plotElecPotentials")
    
    ### Saving results into a Matlab-like object
    logger.info("* Saving results")

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
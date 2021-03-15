# -*- coding: utf-8 -*-
"""
Main script that call model estimation for each combination patient/session/run.

Created on Thu Mar 11 14:41:26 2021

@author: caroline-pinte
"""

# Imports
from pred_NF_from_eeg_fmri_1model_AVC import pred_NF_from_eeg_fmri_1model_AVC
import scipy.io as sio

# Data
# patients = ['P002','P003','P004','P015','P017']
# sessions = ['S1s1','S1s2','S2s1','S3s1']
# learn_runs = ['NF1','NF2','NF3']
# test_runs = learn_runs

# Debug
patients = ['P002']
sessions = ['S1s1']
learn_runs = ['NF1']
test_runs = learn_runs

for p in patients :
    for s in sessions :
        for l in learn_runs :
            for t in test_runs :
                # Paths to load/save 
                data_path = "C:/Users/cpinte/Documents/Data/Patients/"
                res_path = "C:/Users/cpinte/Documents/Results_Python/Res_{}_s{}_l{}_t{}/Res_{}_s{}_l{}_t{}.mat".format(p,s,l,t,p,s,l,t)
                # Call model estimation
                Res = pred_NF_from_eeg_fmri_1model_AVC(data_path, p, s, l, t,'fmri')              
                # Save results object
                sio.savemat(res_path, {'Res':[Res]}) # import into Matlab with : data = load(res_path); Res = [data.Res{:}];
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
15. Single sub analysis: Source reconstruction using template MRI
https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html#sphx-glr-auto-tutorials-forward-35-eeg-no-mri-py
https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

==============================================
Imports the data that has been pre-processed in fieldtrip (ft) and converts into
MNE compatable epochs structure

"""  




import os.path as op

import mne
from mne.parallel import parallel_func
from mne.channels.montage import get_builtin_montages
from warnings import warn
from pymatreader import read_mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from scipy import signal
from scipy import stats 
from scipy.linalg import norm
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse 
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc

import config_for_gogait
                    
def find_closest(arr, val):
    idx = np.abs(arr - val).argmin()
    return arr[idx]

n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# version_list = ['GOODremove','CHANremove']
version_list = ['CHANremove']

event_type = ['target']

orientation = 'varyDipole' #  ['fixDipole', 'varyDipole' ]

method = "dSPM"


# ampliNormalization = ['AmpliNormPerCondi', 'AmpliNormAccCondi', 'AmpliActual']
ampliNormalization = 'AmpliActual'

# event_type = ['target']
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu']
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    
    if method == 'dSPM':
        norm_kind = norm_kind[1] # selecting the normVec [1] option for dSPM  
    elif method == 'MNE':
        norm_kind = norm_kind[2] # selecting the norm_zscored [2] option for MNE
    
    condi_name = ['GOc', 'GOu', 'NoGo']  
    ncondi = len(condi_name)
          
    # t1min = 0.100; t1max = 0.130
    # t2min = 0.150; t2max = 0.200
    # t3min = 0.200; t3max = 0.300
    # t4min = 0.350; t4max = 0.450
    
    # t1min = 0.075; t1max = 0.125
    # t2min = 0.135; t2max = 0.200
    # t3min = 0.225; t3max = 0.340
    # t4min = 0.360; t4max = 0.470
    
    # for ppt on 08/03/2024
    # t1min = 0.136; t1max = 0.164
    t1min = 0.160; t1max = 0.180
    t2min = 0.300; t2max = 0.350
    t3min = 0.370; t3max = 0.450
    
    for veri, version in enumerate(version_list):
        
        ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
        # estimate the num of time samples per condi/ISI to allocate numpy array
        tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
        tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
        n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
        n_chs = 132
        n_verticies = 20484
        n_vectors = 3
        ncondi = len(condi_name)  # nc = 3
        evoked_array_all_sub = np.ones([n_subs, n_chs, n_samples_esti])*np.nan
        stc_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_verticies, n_samples_esti])*np.nan
        
        # goal: normalization across condition for each subject 
        # for each sub, collect all the 3 condi
        for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
              
            for ci, condi in enumerate(condi_name): 
           
                print("Processing subject: %s" % subject)
            
                eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                     
                if method == 'MNE' and orientation == 'fixDipole' : 
                    extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stc' +'_' + version 
                elif method == 'dSPM' and orientation == 'fixDipole' :  
                    extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stc' +'_' + version +'_' + method
                elif method == 'MNE' and orientation == 'varyDipole' : 
                    extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
                elif method == 'dSPM' and orientation == 'varyDipole' :  
                    extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
                
                
                stc_fname_array_in = op.join(eeg_subject_dir_GOODremove,
                                     config_for_gogait.base_fname_npy.format(**locals()))
                print("input: ", stc_fname_array_in)
                 
                stc_data_in = np.load(stc_fname_array_in)
                
                if norm_kind == 'vector':
                    stc_data_per_sub = stc_data_in.copy()
                    
                elif norm_kind == 'normVec': 
                    stc_data_in_norm = norm(stc_data_in, axis = 1)
                    stc_data_per_sub = stc_data_in_norm.copy()
                    #stc_avg_sub = np.mean(stc_data_in_norm, axis = 0)
                elif norm_kind == 'normVec_zsc':
                    stc_data_in_norm = norm(stc_data_in, axis = 1)
                    tmin = 0.2 # pre-stim duration
                    pre_stim_sample_size = int(sfreq * tmin )
                    stc_baseline = stc_data_in_norm[:,0:pre_stim_sample_size]
                    stc_baseline_mean = np.mean(stc_baseline, axis = 1) 
                    stc_baseline_std = np.std(stc_baseline, axis = 1) 
                    num_times_pts = np.shape(stc_data_in_norm)[1] 
                    stc_baseline_mean = np.expand_dims(stc_baseline_mean, axis=1)
                    stc_baseline_std = np.expand_dims(stc_baseline_std, axis=1)
                    mu = np.repeat(stc_baseline_mean,num_times_pts,axis = 1)
                    sig = np.repeat(stc_baseline_std,num_times_pts,axis = 1)
                    stc_data_in_norm_zscored =  (stc_data_in_norm - mu)/sig
                    stc_data_per_sub = stc_data_in_norm_zscored.copy()
                
                # store condi in dim1, vertices in dim2, time in dim3 
                stc_data_per_sub_exp_dim = np.expand_dims(stc_data_per_sub, axis = 0) 
                if ci == 0:
                    stc_data_per_sub_all_condi =  stc_data_per_sub_exp_dim
                else:
                    stc_data_per_sub_all_condi = np.vstack((stc_data_per_sub_all_condi, stc_data_per_sub_exp_dim))                     
                    
            if ampliNormalization == 'AmpliActual': # store subs in dim1, condi in dim2, vertices in dim3, time in dim4
                stc_data_all_sub_all_condi[sub_num,:,:,:] = stc_data_per_sub_all_condi.copy()                      
            elif ampliNormalization == 'AmpliNormAccCondi': ### normalization at single subject across condition, space, time
                tmax = 0.2 + 0.5 # pre-stim + post-stim duration
                sample_size = int(sfreq * tmax )
                # normalize across condition, vertices, time (upto 0.5 s)
                minSTC = np.min(np.min(np.min(stc_data_per_sub_all_condi[:,:,0:sample_size], axis = 2), axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi))
                maxSTC = np.max(np.max(np.max(stc_data_per_sub_all_condi[:,:,0:sample_size], axis = 2), axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi))
                stc_data_per_sub_all_condi_minmax = (stc_data_per_sub_all_condi - minSTC)/(maxSTC - minSTC)
                stc_data_all_sub_all_condi[sub_num,:,:,:] = stc_data_per_sub_all_condi_minmax.copy()
            elif ampliNormalization == 'AmpliNormPerCondi': ### normalization at single subject across condition, space, time
                tmax = 0.2 + 0.5 # pre-stim + post-stim duration
                sample_size = int(sfreq * tmax )
                for ci1 in range(3):
                    # normalize across vertices, time (upto 0.5 s)
                    minSTC = np.min(np.min(stc_data_per_sub_all_condi[ci1,:,0:sample_size], axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi[ci1]))
                    maxSTC = np.max(np.max(stc_data_per_sub_all_condi[ci1,:,0:sample_size], axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi[ci1]))
                    stc_data_per_sub_all_condi_minmax = (stc_data_per_sub_all_condi[ci1] - minSTC)/(maxSTC - minSTC)
                    stc_data_all_sub_all_condi[sub_num,ci1,:,:] = stc_data_per_sub_all_condi_minmax.copy()
         
        ## averaging across subjects
        stc_avg_data_all_condi = np.mean(stc_data_all_sub_all_condi, axis = 0)
            
        #%% plot the sources  
        report = mne.Report()
        fs_dir = fetch_fsaverage(verbose=True)
        template_subjects_dir = op.dirname(fs_dir)
        
        # The files live in:
        template_subject = "fsaverage"
        
        time_points_kind = ['T1', 'T2', 'T3']
        views = ['latView','medView','venView']
        contrast_kind = ['GOu_GOc', 'NoGo_GOu']  
        
        nT = len(time_points_kind)
        nView = len(views)
        ncondi = len(contrast_kind)
        
        
        ## Parcellation used: Human connectome project - multimodal parcellation 1 (HCPMMP1) 
        ## Source: Glasser et al 2016; https://www.nature.com/articles/nature18933
        ## MNE: https://mne.tools/stable/auto_examples/visualization/parcellation.html#sphx-glr-auto-examples-visualization-parcellation-py
        
        for ci, contrast in enumerate(contrast_kind): # contrasts
            for vi, vw in enumerate(views):
                for ti, time_pos in enumerate(time_points_kind): 
            
                    extension = 'stc_funclabel_evkContrast_' + contrast + '_'+ vw +'_' + time_pos + "-lh.label"
                    fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                         config_for_gogait.base_fname_generic.format(**locals()))
                    
                    print("importing label: ", fname)
                    LH_label = mne.read_label(fname)
                
                    extension = 'stc_funclabel_evkContrast_' + contrast + '_'+ vw +'_' + time_pos + "-rh.label"
                    fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                         config_for_gogait.base_fname_generic.format(**locals()))
                    
                    print("importing label: ", fname)
                    RH_label = mne.read_label(fname)
                    
                    for ci, condi in enumerate(condi_name):              
                        stc_avg_data_per_condi = stc_avg_data_all_condi[ci,:,:]
                        vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                        stc_per_condi = mne.SourceEstimate(stc_avg_data_per_condi, vertices = vertno,
                                                  tstep = 1/sfreq, tmin = - 0.2,
                                                  subject = 'fsaverage')   
                       
                        ## labelling separately for left and right hemispheres    
                        #     label_name_hemi = ["lh.BA4a", "rh.BA4a"]
                            
                        for hemi in np.arange(2): # left: hemi = 0; right: hemi = 1
                            if hemi == 0: # left hemisphere
                                label = LH_label
                            else:
                                label = RH_label
                                
                            stc_per_condi_label = stc_per_condi.in_label(label)
                            stc_label_data = stc_per_condi_label.data
                            stc_label_data = np.expand_dims(stc_label_data, axis = 0)
                        
                            if hemi == 0: # left hemisphere
                                if ci == 0:
                                    stc_label_data_condi_LH = stc_label_data
                                else:
                                    stc_label_data_condi_LH = np.vstack((stc_label_data_condi_LH, stc_label_data))    
                                    
                            else: # right hemisphere
                                if ci == 0:
                                    stc_label_data_condi_RH = stc_label_data
                                else:
                                    stc_label_data_condi_RH = np.vstack((stc_label_data_condi_RH, stc_label_data))
                        
                                       
                    #% plotting the ROI curves    
                    fig, ax = plt.subplots(2, 1, layout='constrained')
                    for hemi in np.arange(2): # left: hemi = 0; right: hemi = 1
                        font = {'family': 'serif',
                            'color':  'darkred',
                            'weight': 'normal',
                            'size': 14,
                            }
                        if hemi == 0:
                            hemis = 'LH'
                            stc_label_data_condi = stc_label_data_condi_LH
                            ax[0].plot(stc_per_condi.times, np.mean(stc_label_data_condi[0,:,:], axis = 0).T,
                                       linewidth = 3, color = 'xkcd:lightgreen', label = 'GOc' )
                            ax[0].plot(stc_per_condi.times, np.mean(stc_label_data_condi[1,:,:], axis = 0).T, 
                                       linewidth = 3, color = 'xkcd:salmon', label = 'GOu' )
                            ax[0].plot(stc_per_condi.times, np.mean(stc_label_data_condi[2,:,:], axis = 0).T,
                                       linewidth = 3, color = 'xkcd:grey', label = 'NoGo' )
                            ax[0].axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
                            ax[0].set_xlabel('time (s)', fontdict=font)
                            ax[0].set_ylabel('a.u.', fontdict=font)
                            # ax[0].set_yticks([0,2,4,6], labels=[0,2,4,6])
                            ax[0].tick_params(axis='x', labelsize=14)
                            ax[0].tick_params(axis='y', labelsize=14)
                            ax[0].set_title('LH', fontdict=font)
                            # ax[0].set_ylim([0,0.75])
                            ax[0].axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
                            ax[0].axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
                            ax[0].axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
                            # ax[0].axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
                        else:
                            hemis = 'RH'
                            stc_label_data_condi = stc_label_data_condi_RH
                            ax[1].plot(stc_per_condi.times, np.mean(stc_label_data_condi[0,:,:], axis = 0).T, 
                                       linewidth = 3, color = 'xkcd:lightgreen', label = 'GOc' )
                            ax[1].plot(stc_per_condi.times, np.mean(stc_label_data_condi[1,:,:], axis = 0).T, 
                                       linewidth = 3, color = 'xkcd:salmon', label = 'GOu' )
                            ax[1].plot(stc_per_condi.times, np.mean(stc_label_data_condi[2,:,:], axis = 0).T,
                                       linewidth = 3, color = 'xkcd:grey', label = 'NoGo' )
                            ax[1].axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
                            ax[1].set_xlabel('time (s)', fontdict=font)
                            ax[1].set_ylabel('a.u.', fontdict=font)
                            # ax[1].set_yticks([0,2,4,6], labels=[0,2,4,6])
                            ax[1].tick_params(axis='x', labelsize=14)
                            ax[1].tick_params(axis='y', labelsize=14)
                            ax[1].legend(fontsize=12, loc='upper left')
                            ax[1].set_title('RH', fontdict=font)
                            # ax[1].set_ylim([0,0.75])
                            ax[1].axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
                            ax[1].axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
                            ax[1].axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
                            # ax[1].axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
                                
                            report.add_figure(fig, title =  contrast + '_'+ vw +'_' + time_pos, replace = True)
                            plt.close('all') 
                    
        
        # finally saving the report after the for subject loop ends.     
        print('Saving the reports to disk')  
        report.title = 'Group (n = ' + str(n_subs) + ') avg stc time plots for ROIs evkContrast Panel at_' + evnt+ ': '+ version + '_' + orientation + '_' + method + '_' + ampliNormalization
        extension = 'group_stc_ROI_timePlots_evkContrast_panel'
        report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
        report.save(report_fname+'_at_'+ evnt +'_' + version+ '_' + orientation + '_' + method + '_' + ampliNormalization +'_ppt.html', overwrite=True)            
          
   
                    
              
       
                
            
                
        



    
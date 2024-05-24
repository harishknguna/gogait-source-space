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

# method = "dSPM"
method = 'dSPM'

# ampliNormalization = ['AmpliNormPerCondi', 'AmpliNormAccCondi', 'AmpliActual']
ampliNormalization = 'AmpliNormPerCondi'

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
    
    t1min = 0.075; t1max = 0.125
    t2min = 0.135; t2max = 0.200
    t3min = 0.225; t3max = 0.340
    t4min = 0.360; t4max = 0.470
    
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
                     
                print('  reading the stc numpy array from disk')
               
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
         
            # averaging stc-norm across subjects
            # stc_data_avg_sub_all_condi = np.mean(stc_data_all_sub_all_condi, axis = 0)
            
      
        
        
        #%% Plot the contrast 
        ## Compute and plot the time averaged contrast maps 
        ## do the contrast per individual and then take average across individuals      
        
        contrast_kind = ['GOu_GOc', 'NoGo_GOu']
        time_points_kind = ['T1', 'T2', 'T3', 'T4']
        times  =  np.around(np.arange(-tsec_start,tsec_end,1/sfreq),3)
        report = mne.Report()
                 
        for ti, time_pos in enumerate(time_points_kind): 
            print("for timewindow: %s" % time_pos)       
    
            for ci2, contrast in enumerate(contrast_kind):
                
                if time_pos == 'T1': 
                    tmin = t1min 
                    tmax = t1max 
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                    print('plotting T1 activations for ' + contrast)
                elif time_pos == 'T2':
                    tmin = t2min 
                    tmax = t2max 
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                    print('plotting T2 activations for ' + contrast)
                elif time_pos == 'T3':
                    tmin = t3min 
                    tmax = t3max 
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                    print('plotting T3 activations for ' + contrast)
                elif time_pos == 'T4':
                    tmin = t4min 
                    tmax = t4max 
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                    print('plotting T4 activations for ' + contrast)
                
                tmin_ind = np.where(times == find_closest(times, tmin))[0][0]
                tmax_ind = np.where(times == find_closest(times, tmax))[0][0]
               
                if contrast == 'GOu_GOc':
                    #before: subs in dim1, condi in dim2, vertices in dim3, time in dim4  
                    #after: subs in dim1, vertices in dim2, time in dim3             
                    X1 = np.mean(stc_data_all_sub_all_condi[:,1,:,tmin_ind:tmax_ind],axis =2)  # all subs
                    X2 = np.mean(stc_data_all_sub_all_condi[:,0,:,tmin_ind:tmax_ind],axis =2)  # all subs
                
                elif contrast == 'NoGo_GOu': 
                    X1 = np.mean(stc_data_all_sub_all_condi[:,2,:,tmin_ind:tmax_ind],axis =2)  # all subs
                    X2 = np.mean(stc_data_all_sub_all_condi[:,1,:,tmin_ind:tmax_ind],axis =2)  # all subs
                
                stc_contrast_data =  np.mean(X1 - X2, axis = 0) # averaged across subs
                vmax_mean = np.max(stc_contrast_data)
                vmin_mean = np.min(stc_contrast_data)
                vabsmax_mean = np.max([np.abs(vmax_mean),np.abs(vmin_mean)])
                
                
                # vrange_mean = np.max(stc_contrast_data) - np.min(stc_contrast_data)
                # vmin_mean = 0.0 * vrange_mean + np.min(stc_contrast_data) # 0.50
                # vmid_mean = 0.50 * vrange_mean + np.min(stc_contrast_data) # 0.60
                # vmax_mean = 0.95 * vrange_mean + np.min(stc_contrast_data) # 0.90
                
                stc_contrast_data_std =  np.std(X1 - X2, axis = 0) # std across subs
                vmax_std = np.max(stc_contrast_data_std)
                vmin_std = np.min(stc_contrast_data_std)
                vabsmax_std = np.max([np.abs(vmax_std),np.abs(vmin_std)])
                
                # vrange_std = np.max(stc_contrast_data) - np.min(stc_contrast_data)
                # vmin_std = 0.0 * vrange_std + np.min(stc_contrast_data_std) # 0.50
                # vmid_std = 0.50 * vrange_std + np.min(stc_contrast_data_std) # 0.60
                # vmax_std = 0.95 * vrange_std + np.min(stc_contrast_data_std) # 0.90
                    
                # elif contrast == 'NoGo_GOc': 
                #     X1 = np.mean(stc_data_all_sub_all_condi[:,2,:,tmin_ind:tmax_ind],axis =2) # all subs
                #     X2 = np.mean(stc_data_all_sub_all_condi[:,0,:,tmin_ind:tmax_ind],axis =2) # all subs
                #     stc_contrast_data =  np.mean(X1 - X2, axis = 0) # averaged across subs
                    
                    
                print("plotting the contrast: %s" % contrast)
                vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                stc_contrast = mne.SourceEstimate(stc_contrast_data, vertices = vertno,
                                          tstep = 1/sfreq, tmin = 0,
                                          subject = 'fsaverage') 
                
                stc_contrast_std = mne.SourceEstimate(stc_contrast_data_std, vertices = vertno,
                                          tstep = 1/sfreq, tmin = 0,
                                          subject = 'fsaverage')
                
               
                
                # Performing the paired sample t-test 
                t_stats, pval = stats.ttest_rel(X1, X2) 
                
                pval_log = - np.log10(pval)
                
                stc_tstats = mne.SourceEstimate(t_stats, vertices = vertno,
                                          tstep = 1/sfreq, tmin = 0,
                                          subject = 'fsaverage')
                
                stc_pval = mne.SourceEstimate(pval_log, vertices = vertno,
                                          tstep = 1/sfreq, tmin = 0,
                                          subject = 'fsaverage')
                       
                #% PLOTIING the STC MAPS: contrast, t-maps, and p-value
                
                for plotType in range(4):
                    if plotType == 0:
                        stc_type = stc_contrast.copy()
                        clim=dict(kind="value", pos_lims =  (0, 0.50 * vabsmax_mean, 0.95 * vabsmax_mean)) # in values
                        # use deault colormap with +/- range
                    elif plotType == 1:
                         stc_type = stc_contrast_std.copy() #stc_pval.copy()
                         clim=dict(kind="value", pos_lims =  (0, 0.50 * vabsmax_std, 0.95 * vabsmax_std)) # in values
                         colormap = 'BrBG_r' # colormap with +/- ve range
                    elif plotType == 2:
                        stc_type = stc_tstats.copy()
                        # clim=dict(kind="percent", pos_lims = (60, 90, 95)) # in percentile
                        clim=dict(kind="value", pos_lims = (2.07, 3.79, 5.69)) # in values
                        colormap = 'cool'  # colormap with +/- ve range
                        # t_stats and p_values are matched !!
                    elif plotType == 3:
                        stc_type = stc_pval.copy() 
                        # clim=dict(kind="value", lims = (-0.05, -0.01, -0.001)) 
                        clim=dict(kind="value", lims = (1.3, 3, 5)) # in values
                        colormap = 'YlGn'  # colormap with only +ve range 
                        # t_stats and p_values are matched !!
                        
                    # LATERAL VIEW
                    #% SELECTING ROIs and plotting them
                    labels = mne.read_labels_from_annot(template_subject, 'aparc.a2009s', 
                                                         subjects_dir = template_subjects_dir)
                    
                    wt = 1000
                    ht = 500
                    if plotType == 0: # don't mention colormap (use default) for contrast type 
                        stc_fig1 =  stc_type.plot(
                                            views=["lat"],
                                            hemi= "split",
                                            smoothing_steps=7, 
                                            size=(wt, ht),
                                            view_layout = 'vertical',
                                            time_viewer=True,
                                            show_traces=False,
                                            colorbar=True,
                                            clim=clim
                                    )  
                    else: #plotType == 1 or plotType == 2: 
                        stc_fig1 =  stc_type.plot(
                                            colormap = colormap,
                                            views=["lat"],
                                            hemi= "split",
                                            smoothing_steps=7, 
                                            size=(wt, ht),
                                            view_layout = 'vertical',
                                            time_viewer=True,
                                            show_traces=False,
                                            colorbar=True,
                                            clim=clim
                                    )  
                    
                    # LH
                    if time_pos == 'T1':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif  time_pos == 'T2':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif time_pos == 'T3':
                        my_label= 'S_occipital_ant' #59 Ante. occiptial sulcus and pre-occipital notch
                    elif  time_pos == 'T4':
                        my_label= 'S_front_inf' #52 Inferior frontal sulcus
                    
                    label_lh = [label for label in labels if label.name == my_label+"-lh"][0]
                   
                    # show both labels
                    stc_fig1.add_label(label_lh, borders=True, color='k')
                    
                    # RH
                    if time_pos == 'T1':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif  time_pos == 'T2':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif time_pos == 'T3':
                        my_label= 'S_occipital_ant' #59 Ante. occiptial sulcus and pre-occipital notch
                    elif  time_pos == 'T4':
                        my_label= 'S_front_inf' #52 Inferior frontal sulcus
                    
                    label_rh = [label for label in labels if label.name == my_label+"-rh"][0]
                   
                    # show both labels
                    stc_fig1.add_label(label_rh, borders=True, color='k')
                    screenshot1 = stc_fig1.screenshot()
                    
            #%  MEDIAL VIEW
                    if plotType == 0: # don't mention colormap (use default) for contrast type        
                        stc_fig2 =  stc_type.plot(
                            
                                            views=["med"],
                                            hemi= "split",
                                            smoothing_steps=7, 
                                            size=(wt, ht),
                                            view_layout = 'vertical',
                                            time_viewer=False,
                                            show_traces=False,
                                            colorbar=False,
                                            clim=clim
                                        )  
                    else: #plotType == 1 or plotType == 2:  
                        stc_fig2 =  stc_type.plot(
                                            colormap = colormap,
                                            views=["med"],
                                            hemi= "split",
                                            smoothing_steps=7, 
                                            size=(wt, ht),
                                            view_layout = 'vertical',
                                            time_viewer=False,
                                            show_traces=False,
                                            colorbar=False,
                                            clim=clim
                                        ) 
                        
                    
                    # LH
                    if time_pos == 'T1':
                        my_label= 'G_and_S_cingul-Mid-Post' #8 middle poste cingu gyrus and sulcus pMCC
                    elif  time_pos == 'T2':
                        my_label= 'S_parieto_occipital' #65 parieto-occipital sulcus (or fissure)
                    elif time_pos == 'T3':
                        my_label= 'G_front_sup' #16 superior frontal gyrus
                    elif  time_pos == 'T4':
                        my_label= 'G_and_S_cingul-Mid-Ant'#7 middle ante cingu gyrus and sulcus aMCC
                    
                    label_lh = [label for label in labels if label.name == my_label+"-lh"][0]
                   
                    # show both labels
                    stc_fig2.add_label(label_lh, borders=True, color='k')
                    
                    # RH
                    if time_pos == 'T1':
                        my_label= 'G_and_S_cingul-Mid-Post' #8 middle poste cingu gyrus and sulcus pMCC
                    elif  time_pos == 'T2':
                        my_label= 'S_parieto_occipital'   #65 parieto-occipital sulcus (or fissure)
                    elif time_pos == 'T3':
                        my_label= 'G_front_sup' #16 superior frontal gyrus
                    elif  time_pos == 'T4':
                        my_label= 'G_and_S_cingul-Mid-Ant' #7 middle ante cingu gyrus and sulcus aMCC
                    
                    label_rh = [label for label in labels if label.name == my_label+"-rh"][0]
                   
                    # show both labels
                    stc_fig2.add_label(label_rh, borders=True, color='k')
                    screenshot2 = stc_fig2.screenshot()
             #% DORSAL VENTRAL VIEW
            
                    if plotType == 0: # don't mention colormap (use default) for contrast type 
                        stc_fig3 =  stc_type.plot(
                                            
                                            views=["dor", "ven"],
                                            hemi= "both",
                                            smoothing_steps=7, 
                                            size=(wt, ht),
                                            view_layout = 'horizontal',
                                            time_viewer=False,
                                            show_traces=False,
                                            colorbar=False,
                                            clim=clim
                                        )  
                    else: # plotType == 1 or plotType == 2:
                        stc_fig3 =  stc_type.plot(
                                            colormap = colormap,
                                            views=["dor", "ven"],
                                            hemi= "both",
                                            smoothing_steps=7, 
                                            size=(wt, ht),
                                            view_layout = 'horizontal',
                                            time_viewer=False,
                                            show_traces=False,
                                            colorbar=False,
                                            clim=clim
                                        )  
                    # LH
                    if time_pos == 'T1':
                        my_label= 'S_central' #45 central sulcus (Rolando's fissure)
                    elif  time_pos == 'T2':
                        my_label= 'S_postcentral' #67 postcentral sulcus 
                    elif time_pos == 'T3':
                        my_label= 'G_precentral' #29 precentral gyrus
                    elif  time_pos == 'T4':
                        my_label= 'S_front_sup' #54 superior frontal sulcus
                    
                    label_lh = [label for label in labels if label.name == my_label+"-lh"][0]
                   
                    # show both labels
                    stc_fig3.add_label(label_lh, borders=True, color='k')
                    
                    # RH
                    if time_pos == 'T1':
                        my_label= 'S_central' #45 central sulcus (Rolando's fissure)
                    elif  time_pos == 'T2':
                        my_label= 'S_postcentral' #67 postcentral sulcus 
                    elif time_pos == 'T3':
                        my_label= 'G_precentral' #29 precentral gyrus
                    elif  time_pos == 'T4':
                        my_label= 'S_front_sup' #54 superior frontal sulcus
                    
                    label_rh = [label for label in labels if label.name == my_label+"-rh"][0]
                   
                    # show both labels
                    stc_fig3.add_label(label_rh, borders=True, color='k')
                    screenshot3 = stc_fig3.screenshot()
                    
                    fig = plt.figure(figsize=(18,6))
                    axes = ImageGrid(fig, (1,1,1), nrows_ncols=(1, 3))
                    for ax, image in zip(axes, [screenshot1, screenshot2, screenshot3]):
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.imshow(image)
                    fig.tight_layout()
                     
                    if plotType == 0:
                        report.add_figure(fig, title= contrast + '_' + timeDur+ '_contra_mean', replace = True)
                    elif plotType == 1:
                        report.add_figure(fig, title= contrast + '_' + timeDur+ '_contra_std', replace = True)
                    elif plotType == 2:
                        report.add_figure(fig, title= contrast + '_' + timeDur+ '_tstats', replace = True)
                    elif plotType == 3:
                        report.add_figure(fig, title= contrast + '_' + timeDur+ '_pvalue', replace = True)
                    
                    stc_fig1.close() 
                    stc_fig2.close()
                    stc_fig3.close()
                            
                        
                    plt.close('all')    
                
                
        # finally saving the report after the for condi loop ends.     
        print('Saving the reports to disk')  
        report.title = 'Group (n = ' + str(n_subs) + ') avg contrast maps at '  + evnt +': '+ version +  '_' + orientation + '_'  + method + '_' + ampliNormalization
        extension = 'group_avg_contrast_maps' 
        report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
        report.save(report_fname+'_at_'+ evnt + '_' + version+  '_' + orientation + '_'  + method + '_' + ampliNormalization +'.html', overwrite=True)                  
          
           
   
   



    
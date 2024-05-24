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
    
    ## GFP based timings
    # t1min = 0.075; t1max = 0.125
    # t2min = 0.135; t2max = 0.200
    # t3min = 0.225; t3max = 0.340
    # t4min = 0.360; t4max = 0.470
    
    ## Ziri et al 2024 (in press)
    t1min = 0.100; t1max = 0.130
    t2min = 0.150; t2max = 0.200
    t3min = 0.200; t3max = 0.300
    t4min = 0.250; t4max = 0.350
    t5min = 0.350; t5max = 0.450
    
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
        stc_data_avg_sub_all_condi = np.mean(stc_data_all_sub_all_condi, axis = 0)
        # std dev stc-norm across subjects
        stc_data_std_sub_all_condi = np.std(stc_data_all_sub_all_condi, axis = 0)
            
        #%% plot the sources  
        ## step 1: find max and min values across 3 condi for each T
        
        group_data_stat = ['average', 'stdDev']
        for datakind in group_data_stat:
            if datakind == 'average': 
                dataSTCarray = stc_data_avg_sub_all_condi.copy()
            elif datakind == 'stdDev':
                dataSTCarray = stc_data_std_sub_all_condi.copy()
                
            time_points_kind = ['T1', 'T2', 'T3', 'T4', 'T5']
     
            stc_min_value_condi = np.ones([len(time_points_kind), len(condi_name)])*np.nan
            stc_max_value_condi = np.ones([len(time_points_kind), len(condi_name)])*np.nan
            
            for ti1, time_pos1 in enumerate(time_points_kind): 
                print("for timewindow: %s" % time_pos1)
                
                for ci1, condi1 in enumerate(condi_name): # run condi to get min/max values across 3 conditions 
                    print("computing Min/max of condition: %s" % condi1)
                    vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                    stc = mne.SourceEstimate(dataSTCarray[ci1,:,:], vertices = vertno,
                                              tstep = 1/sfreq, tmin = - 0.2,
                                              subject = 'fsaverage')  
                    if time_pos1 == 'T1': 
                        tmin = t1min 
                        tmax = t1max 
                                           
                    elif time_pos1 == 'T2':
                        tmin = t2min 
                        tmax = t2max 
                                           
                    elif time_pos1 == 'T3':
                        tmin = t3min 
                        tmax = t3max 
                                            
                    elif time_pos1 == 'T4':
                        tmin = t4min 
                        tmax = t4max 
                    
                    elif time_pos1 == 'T5':
                        tmin = t5min 
                        tmax = t5max 
                                  
                
                    # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
                    stc_cropped = stc.copy()
                    stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
                    stc_min_value_condi[ti1, ci1] = np.min(stc_mean_timepts.data)
                    stc_max_value_condi[ti1, ci1] = np.max(stc_mean_timepts.data)
                    
      
            #% PLOT with common scale
            
            report = mne.Report()
            stc_min_value_T = np.min(stc_min_value_condi, axis = 1).flatten()
            stc_max_value_T = np.max(stc_max_value_condi, axis = 1).flatten()
             
            for ti2, time_pos2 in enumerate(time_points_kind): 
                print("for timewindow: %s" % time_pos2)  
                for ci2, condi2 in enumerate(condi_name):                 
                    print("plotting the condition: %s" % condi2)
                    vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                    stc = mne.SourceEstimate(dataSTCarray[ci2,:,:], vertices = vertno,
                                              tstep = 1/sfreq, tmin = - 0.2,
                                              subject = 'fsaverage')                
                           
                    # plotting the snapshots at 3 different time zones
                    # # time_points_kind = ['early', 'mid', 'late']
                    # time_points_kind = ['early']
            
                    if time_pos2 == 'T1': 
                        tmin = t1min 
                        tmax = t1max 
                        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                        print('plotting T1 activations for ' + condi2)
                    elif time_pos2 == 'T2':
                        tmin = t2min 
                        tmax = t2max 
                        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                        print('plotting T2 activations for ' + condi2)
                    elif time_pos2 == 'T3':
                        tmin = t3min 
                        tmax = t3max 
                        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                        print('plotting T3 activations for ' + condi2)
                    elif time_pos2 == 'T4':
                        tmin = t4min 
                        tmax = t4max 
                        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                        print('plotting T4 activations for ' + condi2)
                
                    # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
                    stc_cropped = stc.copy()
                    stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
                 
                    #% LATERAL VIEW
                    #% SELECTING ROIs and plotting them
                    labels = mne.read_labels_from_annot(template_subject, 'aparc.a2009s', 
                                                         subjects_dir = template_subjects_dir)
                    
                    # clim=dict(kind="percent", lims = (60, 90, 95)), # in percentile
                    vrange = stc_max_value_T[ti2] - stc_min_value_T[ti2]
                    vmin = 0.60 * vrange + stc_min_value_T[ti2] # 0.0
                    vmid = 0.70 * vrange + stc_min_value_T[ti2] # 0.50
                    vmax = 0.90 * vrange + stc_min_value_T[ti2] # 0.95
                    clim = dict(kind="value", lims=[vmin, vmid, vmax])
                    
                    wt = 1000
                    ht = 500
                    stc_fig1 =  stc_mean_timepts.plot(
                                        views=["lat"],
                                        hemi= "split",
                                        smoothing_steps=7, 
                                        size=(wt, ht),
                                        view_layout = 'vertical',
                                        time_viewer=True,
                                        show_traces=False,
                                        colorbar=True,
                                        clim = clim # in values
                                    )  
                    
                    # LH
                    if time_pos2 == 'T1':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif  time_pos2 == 'T2':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif time_pos2 == 'T3':
                        my_label= 'S_occipital_ant' #59 Ante. occiptial sulcus and pre-occipital notch
                    elif  time_pos2 == 'T4':
                        my_label= 'S_front_inf' #52 Inferior frontal sulcus
                    
                    label_lh = [label for label in labels if label.name == my_label+"-lh"][0]
                   
                    # show both labels
                    stc_fig1.add_label(label_lh, borders=True, color='k')
                    
                    # RH
                    if time_pos2 == 'T1':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif  time_pos2 == 'T2':
                        my_label= 'Lat_Fis-post' #41 Posterior ramus (or segment) of the lateral sulcus (or fissure)
                    elif time_pos2 == 'T3':
                        my_label= 'S_occipital_ant' #59 Ante. occiptial sulcus and pre-occipital notch
                    elif  time_pos2 == 'T4':
                        my_label= 'S_front_inf' #52 Inferior frontal sulcus
                    
                    label_rh = [label for label in labels if label.name == my_label+"-rh"][0]
                   
                    # show both labels
                    stc_fig1.add_label(label_rh, borders=True, color='k')
                    screenshot1 = stc_fig1.screenshot()
                    
                    #%  MEDIAL VIEW        
                    stc_fig2 =  stc_mean_timepts.plot(
                                        views=["med"],
                                        hemi= "split",
                                        smoothing_steps=7, 
                                        size=(wt, ht),
                                        view_layout = 'vertical',
                                        time_viewer=False,
                                        show_traces=False,
                                        colorbar=False,
                                        clim = clim # in values
                                    )  
                    
                    # LH
                    if time_pos2 == 'T1':
                        my_label= 'G_and_S_cingul-Mid-Post' #8 middle poste cingu gyrus and sulcus pMCC
                    elif  time_pos2 == 'T2':
                        my_label= 'S_parieto_occipital' #65 parieto-occipital sulcus (or fissure)
                    elif time_pos2 == 'T3':
                        my_label= 'G_front_sup' #16 superior frontal gyrus
                    elif  time_pos2 == 'T4':
                        my_label= 'G_and_S_cingul-Mid-Ant'#7 middle ante cingu gyrus and sulcus aMCC
                    
                    label_lh = [label for label in labels if label.name == my_label+"-lh"][0]
                   
                    # show both labels
                    stc_fig2.add_label(label_lh, borders=True, color='k')
                    
                    # RH
                    if time_pos2 == 'T1':
                        my_label= 'G_and_S_cingul-Mid-Post' #8 middle poste cingu gyrus and sulcus pMCC
                    elif  time_pos2 == 'T2':
                        my_label= 'S_parieto_occipital'   #65 parieto-occipital sulcus (or fissure)
                    elif time_pos2 == 'T3':
                        my_label= 'G_front_sup' #16 superior frontal gyrus
                    elif  time_pos2 == 'T4':
                        my_label= 'G_and_S_cingul-Mid-Ant' #7 middle ante cingu gyrus and sulcus aMCC
                    
                    label_rh = [label for label in labels if label.name == my_label+"-rh"][0]
                   
                    # show both labels
                    stc_fig2.add_label(label_rh, borders=True, color='k')
                    screenshot2 = stc_fig2.screenshot()
            
                    #%DORSAL VENTRAL VIEW               
                    stc_fig3 =  stc_mean_timepts.plot(
                                        views=["dor", "ven"],
                                        hemi= "both",
                                        smoothing_steps=7, 
                                        size=(wt, ht),
                                        view_layout = 'horizontal',
                                        time_viewer=False,
                                        show_traces=False,
                                        colorbar=False,
                                        clim = clim # in values
                                    )  
                    # LH
                    if time_pos2 == 'T1':
                        my_label= 'S_central' #45 central sulcus (Rolando's fissure)
                    elif  time_pos2 == 'T2':
                        my_label= 'S_postcentral' #67 postcentral sulcus 
                    elif time_pos2 == 'T3':
                        my_label= 'G_precentral' #29 precentral gyrus
                    elif  time_pos2 == 'T4':
                        my_label= 'S_front_sup' #54 superior frontal sulcus
                    
                    label_lh = [label for label in labels if label.name == my_label+"-lh"][0]
                   
                    # show both labels
                    stc_fig3.add_label(label_lh, borders=True, color='k')
                    
                    # RH
                    if time_pos2 == 'T1':
                        my_label= 'S_central' #45 central sulcus (Rolando's fissure)
                    elif  time_pos2 == 'T2':
                        my_label= 'S_postcentral' #67 postcentral sulcus 
                    elif time_pos2 == 'T3':
                        my_label= 'G_precentral' #29 precentral gyrus
                    elif  time_pos2 == 'T4':
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
                    
                    report.add_figure(fig, title= condi2 + '_' + timeDur, replace = True)
               
                    stc_fig1.close() 
                    stc_fig2.close()
                    stc_fig3.close()
                        
                    
                    plt.close('all')    
            
            
            # finally saving the report after the for condi loop ends.     
            print('Saving the reports to disk')  
            report.title = 'Group (n = ' + str(n_subs) + ') STC_'+  datakind +'_' + evnt + ': ' + version + '_' + orientation + '_' + method + '_' + ampliNormalization
            extension = 'group_stc_'+  datakind +'_maps'
            report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
            report.save(report_fname+'_at_'+ evnt + '_' + version+ '_' + orientation + '_'  + method + '_' + ampliNormalization +'_2.html', overwrite=True)                  
              
   
   



    
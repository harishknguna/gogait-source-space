#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
Group level EVOKED CONTRAST 
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

method = "dSPM"  # "MNE" | "dSPM"

# ampliNormalization = ['AmpliNormPerCondi', 'AmpliNormAccCondi', 'AmpliActual']
ampliNormalization = 'AmpliActual'


# event_type = ['target']
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # contrast_kind = ['GOu']
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    
    if method == 'dSPM':
        norm_kind = norm_kind[1] # selecting the normVec [1] option for dSPM  
    elif method == 'MNE':
        norm_kind = norm_kind[2] # selecting the norm_zscored [2] option for MNE
    
    # contrast_kind = ['NoGo_GOc', 'NoGo_GOu', 'GOu_GOc']  
    contrast_kind = ['GOu_GOc', 'NoGo_GOu']  
    ncondi = len(contrast_kind)
          
    # t1min = 0.100; t1max = 0.130
    # t2min = 0.150; t2max = 0.200
    # t3min = 0.200; t3max = 0.300
    # t4min = 0.350; t4max = 0.450
   
    # ## GFP based timings
    # t0min = - 0.200; t0max = 0.0
    # t1min = 0.075; t1max = 0.125
    # t2min = 0.135; t2max = 0.200
    # t3min = 0.225; t3max = 0.340
    # t4min = 0.360; t4max = 0.470
    
    # ## Ziri et al 2024 (submitted)
    # t1min = 0.100; t1max = 0.130
    # t2min = 0.150; t2max = 0.200
    # t3min = 0.200; t3max = 0.300
    # t4min = 0.250; t4max = 0.350
    # t5min = 0.350; t5max = 0.450
    
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
       
        evoked_array_all_sub = np.ones([n_subs, n_chs, n_samples_esti])*np.nan
        stc_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_verticies, n_samples_esti])*np.nan
        
        # goal: normalization across condition for each subject 
        # for each sub, collect all the 3 condi
        for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
              
            for ci, contrast in enumerate(contrast_kind): # three contrasts
           
                print("Processing subject: %s" % subject)
            
                eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                     
                print('  reading the stc numpy array from disk')
                
                if method == 'MNE' and orientation == 'fixDipole' : 
                    extension = contrast +'_' + event_type[ei] + '_' +'stc' +'_' + version 
                elif method == 'dSPM' and orientation == 'fixDipole' :  
                    extension = contrast  +'_' + event_type[ei] + '_' +'stc' +'_' + version +'_' + method
                elif method == 'MNE' and orientation == 'varyDipole' : 
                    extension = contrast  +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
                elif method == 'dSPM' and orientation == 'varyDipole' :  
                    extension = contrast  +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
                
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
            
        #%% saving the STCs (grand mean and sd) for each contra as numpy array.
        for ci, contrast in enumerate(contrast_kind):
            print('\n Writing the stcGAVG to disk: %s'  % contrast)
            extension = contrast_kind[ci] +'_' + event_type[ei] + '_' +'stcGAVG' +'_evkContrast_' + version +'_' + method +'_' + ampliNormalization 
            stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
                                  config_for_gogait.base_fname_avg_npy.format(**locals()))
            print("Output: ", stc_fname_array)
            stc_avg_data = stc_data_avg_sub_all_condi[ci,:,:]
            np.save(stc_fname_array, stc_avg_data)
            
            print('\n Writing the stcGSTD to disk: %s'  % contrast)
            extension = contrast_kind[ci] +'_' + event_type[ei] + '_' +'stcGSTD' +'_evkContrast_' + version +'_' + method +'_' + ampliNormalization 
            stc_fname_array = op.join( config_for_gogait.eeg_dir_GOODremove,
                                  config_for_gogait.base_fname_avg_npy.format(**locals()))
            print("Output: ", stc_fname_array)
            stc_std_data = stc_data_std_sub_all_condi[ci,:,:]
            np.save(stc_fname_array, stc_std_data)
            
        #%% plot the sources  
        ## step 1: find max and min values across 3 condi for each T

datakind = 'average'    
dataSTCarray = stc_data_avg_sub_all_condi.copy()
# time_points_kind = ['T1', 'T2', 'T3', 'T4', 'T5']
time_points_kind = ['T1', 'T2', 'T3']
nT = len(time_points_kind)
 
stc_min_value_condi = np.ones([len(time_points_kind), len(contrast_kind)])*np.nan
stc_max_value_condi = np.ones([len(time_points_kind), len(contrast_kind)])*np.nan

for ti1, time_pos1 in enumerate(time_points_kind): 
    print("for timewindow: %s" % time_pos1)
    
    for ci1, condi1 in enumerate(contrast_kind): # run condi to get min/max values across 3 conditions 
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
                                
        # elif time_pos1 == 'T4':
        #     tmin = t4min 
        #     tmax = t4max 
        
        # elif time_pos1 == 'T5':
        #     tmin = t5min 
        #     tmax = t5max 
                      
    
        # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
        stc_cropped = stc.copy()
        stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
        stc_min_value_condi[ti1, ci1] = np.min(stc_mean_timepts.data)
        stc_max_value_condi[ti1, ci1] = np.max(stc_mean_timepts.data)
        
#%%  plot the figures 
report = mne.Report()
scale = 'timeScale' # 'ptileScale' | 'globalScale' | 'timeScale' | 'condiScale'

## PLOT with common scale time point wise (rows) or condi (col) wise or both
## here time point wise
if scale == 'timeScale':
    stc_min_value_T = np.min(stc_min_value_condi, axis = 1).flatten()
    stc_max_value_T = np.max(stc_max_value_condi, axis = 1).flatten()
## here condi wise
elif scale == 'condiScale':
    stc_min_value_T = np.min(stc_min_value_condi, axis = 0).flatten()
    stc_max_value_T = np.max(stc_max_value_condi, axis = 0).flatten()
## here timepoint and condi wise: globally
elif scale == 'globalScale': 
    stc_min_value_T = np.min(np.min(stc_min_value_condi, axis = 0))
    stc_max_value_T = np.max(np.max(stc_max_value_condi, axis = 0))
elif scale == 'ptileScale': 
    stc_min_value_T = np.nan
    stc_max_value_T = np.nan



for ci2, condi2 in enumerate(contrast_kind):                 
    print("plotting the condition: %s" % condi2)
    
   ## figure to plot brain
   
    fig = plt.figure(figsize=(9,3)) #18 3
    brain_views = 3
    axes = ImageGrid(fig, (1,1,1), nrows_ncols=(brain_views, nT)) #
    figInd = 0
    
    for ti2, time_pos2 in enumerate(time_points_kind): 
        print("for timewindow: %s" % time_pos2)  
        
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
        # elif time_pos2 == 'T4':
        #     tmin = t4min 
        #     tmax = t4max 
        #     timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
        #     print('plotting T4 activations for ' + condi2)
        # elif time_pos2 == 'T5':
        #     tmin = t5min 
        #     tmax = t5max 
        #     timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
        #     print('plotting T5 activations for ' + condi2)
    
        # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
        stc_cropped = stc.copy()
        stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
     
        #% LATERAL VIEW
        
        ## percentile 
        if scale == 'ptileScale':
            vmin = 96 # %tile
            vmid = 97.5 # %tile
            vmax = 99.95 # %tile
            clim=dict(kind="percent", lims = [vmin, vmid, vmax]) # in percentile
        
        ## time wise scale
        elif scale == 'timeScale':
            vrange = stc_max_value_T[ti2] - stc_min_value_T[ti2]
            baseval = stc_min_value_T[ti2]
            vmin = 0.60 * vrange + baseval # 0.0
            vmid = 0.70 * vrange + baseval # 0.50
            vmax = 0.90 * vrange + baseval # 0.95
            clim = dict(kind="value", lims = [vmin, vmid, vmax])
        
        ## condi wise scale
        elif scale == 'condiScale':
            vrange = stc_max_value_T[ci2] - stc_min_value_T[ci2]
            baseval = stc_min_value_T[ci2]
            vmin = 0.60 * vrange + baseval # 0.0
            vmid = 0.70 * vrange + baseval # 0.50
            vmax = 0.90 * vrange + baseval # 0.95
            clim = dict(kind="value", lims = [vmin, vmid, vmax])
        
        ## global scale
        elif scale == 'globalScale': 
            vrange = stc_max_value_T - stc_min_value_T
            baseval = stc_min_value_T
            vmin = 0.60 * vrange + baseval # 0.0
            vmid = 0.70 * vrange + baseval # 0.50
            vmax = 0.90 * vrange + baseval # 0.95
            clim = dict(kind="value", lims = [vmin, vmid, vmax])
        
        clrBar = True
        
        # if ti2 == 4:
        #     clrBar = True
        # else:
        #     clrBar = False
        
        wt = 200 #1000
        ht = 100 #500
        stc_fig1 =  stc_mean_timepts.plot(
                            views=["lat"],
                            hemi= "split",
                            smoothing_steps=7, 
                            size=(wt, ht),
                            view_layout = 'vertical',
                            time_viewer=False,
                            show_traces=False,
                            colorbar= clrBar,
                            background='white',
                            clim = clim, # in values
                            brain_kwargs = dict(surf = 'inflated'),
                            add_data_kwargs = dict(colorbar_kwargs=
                                                   dict(vertical = False,
                                                        n_labels = 3,
                                                        label_font_size = 10,
                                                        width = 0.8, 
                                                        height = 0.2, 
                                                        fmt = '%.2f'
                                                        )
                                                   )
                        )  
        
        
       
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
                            colorbar= False,
                            background='white',
                            clim = clim, # in values
                            brain_kwargs = dict(surf = 'inflated'),
                            
                        )  
        
             
       
        screenshot2 = stc_fig2.screenshot()

        #%DORSAL VENTRAL VIEW     
        ### https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_scalar_bar.html          
        stc_fig3 =  stc_mean_timepts.plot(
                            views=["dor", "ven"],
                            hemi= "both",
                            smoothing_steps=7, 
                            size=(wt, ht),
                            view_layout = 'horizontal',
                            time_viewer= False,
                            show_traces= False,
                            colorbar= False,
                            background='white',
                            clim = clim, # in values
                            brain_kwargs = dict(surf = 'inflated'), 
                            
                                                                                 
                        )  
        
        
               
        screenshot3 = stc_fig3.screenshot()
       
        ax_ind = 0
        for ax, image in zip([axes[figInd + ti2],axes[figInd+ nT+ ti2],axes[figInd + 2*nT + ti2]],
                             [screenshot1, screenshot2, screenshot3]):
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.axis('off')
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(False)
            ax.imshow(image)
            ax_ind = ax_ind + 1
        
        fig.tight_layout()
        
        stc_fig1.close() 
        stc_fig2.close()
        stc_fig3.close()
        
            
    report.add_figure(fig, title= condi2, replace = True)
    plt.close('all')    


# finally saving the report after the for condi loop ends.     
print('Saving the reports to disk')  
report.title = 'Group (n = ' + str(n_subs) + ') STC_evkContrast_'+  datakind +'_' + evnt + ': ' + version + '_' + orientation + '_' + method + '_' + ampliNormalization + '_' + scale
extension = 'group_PANEL_stc_evkContrast_'+  datakind +'_maps'
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt + '_' + version+ '_' + orientation + '_'  + method + '_' + ampliNormalization + '_' + scale + '_ppt.html', overwrite=True)                  
  
   
           


    
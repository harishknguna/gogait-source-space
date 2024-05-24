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

version_list = ['CHANremove'] # 'GOODremove' | 'CHANremove'

method = "dSPM"  # "MNE" | "dSPM"

condi_name = ['GOc', 'GOu', 'NoGo']

event_type = ['target']   # 'target' | 'cue'

orientation = 'varyDipole'  # 'varyDipole' | 'fixed'

#####################################################
time_pos = 'T4' # CHANGE HERE ['T1', 'T2', 'T3', 'T4']
######################################################


for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   

    norm_kind = ['vector','normVec', 'normVec_zsc']  
    
    if method == 'dSPM':
        norm_kind = norm_kind[1] # selecting the normVec [1] option for dSPM  
    elif method == 'MNE':
        norm_kind = norm_kind[2] # selecting the norm_zscored [2] option for MNE
                                            
    
    # condi_name = ['GOc', 'GOu', 'NoGo']
    # time_points_kind = ['T1', 'T2', 'T3', 'T4']
    
    # t1min = 0.100; t1max = 0.130
    # t2min = 0.150; t2max = 0.200
    # t3min = 0.200; t3max = 0.300
    # t4min = 0.350; t4max = 0.450
    
    t1min = 0.075; t1max = 0.125
    t2min = 0.135; t2max = 0.200
    t3min = 0.225; t3max = 0.340
    t4min = 0.360; t4max = 0.470

    for veri, version in enumerate(version_list):
    
        for ci, condi in enumerate(condi_name): 
            
            # creating reports per contrast type
            report = mne.Report()
            print ('single sub STC average across time for ' + condi + ' at ' + evnt)
            
            for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
            
                ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
                # estimate the num of time samples per condi/ISI to allocate numpy array
                tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
                tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
                n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
                n_chs = 132
                n_verticies = 20484
                n_vectors = 3   
                 
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
                                
                #%% plot the sources  
                vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                stc = mne.SourceEstimate(stc_data_per_sub, vertices = vertno,
                                          tstep = 1/sfreq, tmin = - 0.2,
                                          subject = 'fsaverage')                
                       
                # plotting the snapshots at 3 different time zones
                # # time_points_kind = ['early', 'mid', 'late']
                # time_points_kind = ['early']
                #for ind, time_pos in enumerate(time_points_kind): # time position
                    
                if time_pos == 'T1': 
                    tminn= t1min 
                    tmaxx= t1max 
                    timeDur1 = str(int(tminn*1000)) + '_' + str(int(tmaxx*1000)) 
                elif time_pos == 'T2':
                    tminn= t2min 
                    tmaxx= t2max 
                    timeDur2 = str(int(tminn*1000)) + '_' + str(int(tmaxx*1000)) 
                elif time_pos == 'T3':
                    tminn= t3min 
                    tmaxx= t3max 
                    timeDur3 = str(int(tminn*1000)) + '_' + str(int(tmaxx*1000))
                elif time_pos == 'T4':
                    tminn= t4min
                    tmaxx= t4max
                    timeDur4 = str(int(tminn*1000)) + '_' + str(int(tmaxx*1000))
                    
                timeDur = str(int(tminn*1000)) + '_' + str(int(tmaxx*1000))   
                stc_cropped = stc.copy()
                stc_mean_timepts = stc_cropped.crop(tmin= tminn,tmax= tmaxx).mean()
                
                vmax_stc = np.max(stc_mean_timepts.data)
                vmin_stc = np.min(stc_mean_timepts.data)
                vrange = vmax_stc - vmin_stc
                vmin = 0.0 * vrange + vmin_stc
                vmid = 0.50 * vrange + vmin_stc
                vmax = 0.95 * vrange + vmin_stc
                clim = dict(kind="value", lims=[vmin, vmid, vmax])
                
                
                #vabsmax_mean = np.max([np.abs(vmax_mean),np.abs(vmin_mean)])
                
                    
                       
                        # # ## Min–max normalization (-1,1)
                        # # stc_min = np.min(stc_meanTimepts.data)
                        # # stc_max = np.max(stc_meanTimepts.data)
                        # # stc_mean_timepts_minmax = -np.ones(np.shape(stc_meanTimepts)) + 2*(stc_meanTimepts.data - stc_min)/(stc_max - stc_min)
                               
                        # stc_meanTimepts = mne.SourceEstimate(stc_mean_timepts, vertices = vertno,
                        #                           tstep = 1/sfreq, tminn= 0,
                        #                           subject = 'fsaverage')
                        
                        
                        # # ## sanity check
                        # dipole_data = stc_mean_timepts_minmax.copy() 
                        # auto_pts = [np.percentile(dipole_data, 96), 
                        # np.percentile(dipole_data, 97.5), 
                        # np.percentile(dipole_data, 99.95)]
                        
                        
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
                                    clim= clim #dict(kind="percent", lims = (60, 90, 95)), # in percentile
                                )  
                # stc_fig_data = stc_fig.data
                screenshot1 = stc_fig1.screenshot()
                
                stc_fig2 =  stc_mean_timepts.plot(
                                    views=["med"],
                                    hemi= "split",
                                    smoothing_steps=7, 
                                    size=(wt, ht),
                                    view_layout = 'vertical',
                                    time_viewer=False,
                                    show_traces=False,
                                    colorbar=False,
                                    clim= clim #dict(kind="percent", lims = (60, 90, 95)), # in percentile
                                )  
                # stc_fig_data = stc_fig.data
                screenshot2 = stc_fig2.screenshot()
                
                stc_fig3 =  stc_mean_timepts.plot(
                                    views=["dor", "ven"],
                                    hemi= "both",
                                    smoothing_steps=7, 
                                    size=(wt, ht),
                                    view_layout = 'horizontal',
                                    time_viewer=False,
                                    show_traces=False,
                                    colorbar=False,
                                    clim= clim #dict(kind="percent", lims = (60, 90, 95)), # in percentile
                                )  
                # stc_fig_data = stc_fig.data
                screenshot3 = stc_fig3.screenshot()
                
                fig = plt.figure(figsize=(18,6))
                axes = ImageGrid(fig, (1,1,1), nrows_ncols=(1, 3))
                for ax, image in zip(axes, [screenshot1, screenshot2, screenshot3]):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(image)
                fig.tight_layout()
    
                    
                
                report.add_figure(fig, title= subject, replace = True)
               
                stc_fig1.close() 
                stc_fig2.close()
                stc_fig3.close()
                    
                
                plt.close('all')    
                   
            # finally saving the report after the for subject loop ends.     
            print('Saving the reports to disk')  
            report.title = 'single sub STC averaged maps for ' + condi + '_at_'+ evnt +'_' + timeDur + '_' + version +'_' + orientation + '_' + method 
            extension = 'single_sub_stc_averaged_maps_for_' + condi
            report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
            report.save(report_fname +'_at_'+ evnt +'_' + timeDur + '_' + version +'_' + orientation + '_' + method +'.html', overwrite=True)                  
          
   
   



    
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
    
    condi_name = ['GOc', 'GOu', 'NoGo']  
    ncondi = len(condi_name)
    
    contrast_kind = ['GOu_GOc', 'NoGo_GOu']  
    ncontra = len(contrast_kind)
          
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
        stc_data_avg_sub_all_condi = np.ones([ncondi, n_verticies, n_samples_esti])*np.nan
        stc_data_std_sub_all_condi = np.ones([ncondi, n_verticies, n_samples_esti])*np.nan
        stc_data_avg_sub_all_contra = np.ones([ncontra, n_verticies, n_samples_esti])*np.nan
        stc_data_std_sub_all_contra = np.ones([ncontra, n_verticies, n_samples_esti])*np.nan
            
        ## loading the STCs (grand mean and sd) for each condi as numpy array.
        for ci, condi in enumerate(condi_name):
            print('\n Writing the stcGAVG to disk: %s'  % condi)
            extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stcGAVG' +'_' + version +'_' + method +'_' + ampliNormalization 
            stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
                                  config_for_gogait.base_fname_avg_npy.format(**locals()))
            print("Output: ", stc_fname_array)
            stc_data_avg_sub_all_condi[ci,:,:] = np.load(stc_fname_array)
           
            
            print('\n Writing the stcGSTD to disk: %s'  % condi)
            extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stcGSTD' +'_' + version +'_' + method +'_' + ampliNormalization 
            stc_fname_array = op.join( config_for_gogait.eeg_dir_GOODremove,
                                  config_for_gogait.base_fname_avg_npy.format(**locals()))
            print("Output: ", stc_fname_array)
            stc_data_std_sub_all_condi[ci,:,:] = np.load(stc_fname_array)
       
        ## loading the STCs (grand mean and sd) for each contrast as numpy array.  
        for ci, contrast in enumerate(contrast_kind):
            print('\n Writing the stcGAVG to disk: %s'  % contrast)
            extension = contrast_kind[ci] +'_' + event_type[ei] + '_' +'stcGAVG' +'_evkContrast_' + version +'_' + method +'_' + ampliNormalization 
            stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
                                  config_for_gogait.base_fname_avg_npy.format(**locals()))
            print("Output: ", stc_fname_array)
            stc_data_avg_sub_all_contra[ci,:,:] = np.load(stc_fname_array)
           
            
            print('\n Writing the stcGSTD to disk: %s'  % contrast)
            extension = contrast_kind[ci] +'_' + event_type[ei] + '_' +'stcGSTD' +'_evkContrast_' + version +'_' + method +'_' + ampliNormalization 
            stc_fname_array = op.join( config_for_gogait.eeg_dir_GOODremove,
                                  config_for_gogait.base_fname_avg_npy.format(**locals()))
            print("Output: ", stc_fname_array)
            stc_data_std_sub_all_contra[ci,:,:] = np.load(stc_fname_array)
                

#%% reading src of fsaverage
fname_src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(fname_src)


#%%

datakind = 'average'    
dataSTCarray = stc_data_avg_sub_all_contra.copy()   ### consider CONTRAST data for brain plot
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
        
#%  plot the figures 
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
        
        # LATERAL VIEW
        #% SELECTING ROIs and plotting them
        ## step 1:
        labels = mne.read_labels_from_annot(template_subject, 'aparc.a2009s', 
                                             subjects_dir = template_subjects_dir)
        vth_label = vmid #0.50 * vrange + baseval ## vmin | vmid | vmax
        
               
        ### Functional labels 
        stc_with_zeros = stc_mean_timepts.copy()
        data = np.abs(stc_with_zeros.data)
        stc_with_zeros.data[data < vth_label] = 0.0

        # 8.5% of original source space vertices were omitted during forward
        # calculation, suppress the warning here with verbose='error'
        func_labels_all_LH,  func_labels_all_RH = mne.stc_to_label(
            stc_with_zeros,
            src=src,
            smooth=True,
            # subjects_dir=subjects_dir,
            connected=True,
            verbose="error",
        )
        
        ## PLOT figures for LH and RH
        hemi_list = ['LH', 'RH']
        
        for hemi in hemi_list: 
            if hemi == 'LH': 
                n_clus_lab = len(func_labels_all_LH)
            elif hemi == 'RH':
                n_clus_lab = len(func_labels_all_RH)
            
            #% plotting the ROI curves 
            for li in np.arange(n_clus_lab):
                
                if hemi == 'LH': 
                    func_label = func_labels_all_LH[li]
                    hemi_small = 'lh'
                elif hemi == 'RH':
                    func_label = func_labels_all_RH[li]
                    hemi_small = 'rh'
                
                
                fig, axd = plt.subplot_mosaic(
                                        [["screenshot", "timeCourse"]],
                                        layout="constrained", 
                                        figsize=(14,3), dpi=100,
                                        width_ratios=[3.2, 1.8]
                                                                  
                                    )
                                      
                wt = 1000#1000
                ht = 250#500
                stc_fig1 =  stc_mean_timepts.plot(
                                    views=["lat","dor","med", "ven"],
                                    hemi= hemi_small ,#"split",
                                    smoothing_steps=7, 
                                    size=(wt, ht),
                                    view_layout = 'horizontal',
                                    time_viewer=False,
                                    show_traces=False,
                                    colorbar= clrBar,
                                    background='white',
                                    clim = clim, # in values
                                    brain_kwargs = dict(surf = 'inflated'),
                                    add_data_kwargs = dict(colorbar_kwargs=
                                                           dict(vertical = False,
                                                                n_labels = 3,
                                                                label_font_size = 14,
                                                                width = 0.8, 
                                                                height = 0.1, 
                                                                fmt = '%.2f'
                                                                )
                                                           )
                                )  
               
                # show both labels
                stc_fig1.add_label(func_label, borders=True, color='k')
                screenshot1 = stc_fig1.screenshot()
                axd["screenshot"].imshow(screenshot1)
                axd["screenshot"].axis('off')
               
                
                
                ## plotting STC of all condi averaged across the label
                
                for ci, condi in enumerate(condi_name):              
                    stc_avg_data_per_condi = stc_data_avg_sub_all_condi[ci,:,:]  ### consider CONDI data for brain plot
                    vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                    stc_per_condi = mne.SourceEstimate(stc_avg_data_per_condi, vertices = vertno,
                                              tstep = 1/sfreq, tmin = - 0.2,
                                              subject = 'fsaverage')   
                    stc_per_condi_label = stc_per_condi.in_label(func_label)
                    stc_label_data = stc_per_condi_label.data
                    stc_label_data = np.expand_dims(stc_label_data, axis = 0)
                    if ci == 0:
                        stc_label_data_condi = stc_label_data
                    else:
                        stc_label_data_condi = np.vstack((stc_label_data_condi, stc_label_data))    
                        
                
                font = {'family': 'serif',
                    'color':  'darkred',
                    'weight': 'normal',
                    'size': 14,
                    }
                
                #stc_label_data_condi = stc_label_data_condi
                axd["timeCourse"].plot(stc_per_condi.times, np.mean(stc_label_data_condi[0,:,:], axis = 0).T,
                           linewidth = 3, color = 'xkcd:lightgreen', label = 'GOc' )
                axd["timeCourse"].plot(stc_per_condi.times, np.mean(stc_label_data_condi[1,:,:], axis = 0).T, 
                           linewidth = 3, color = 'xkcd:salmon', label = 'GOu' )
                axd["timeCourse"].plot(stc_per_condi.times, np.mean(stc_label_data_condi[2,:,:], axis = 0).T,
                           linewidth = 3, color = 'xkcd:grey', label = 'NoGo' )
                axd["timeCourse"].axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
                axd["timeCourse"].set_xlabel('time (s)', fontdict=font)
                axd["timeCourse"].set_ylabel('a.u.', fontdict=font)
                axd["timeCourse"].set_yticks([0,2,4,6,8,10], labels=[0,2,4,6,8,10])
                axd["timeCourse"].tick_params(axis='x', labelsize=14)
                axd["timeCourse"].tick_params(axis='y', labelsize=14)
                axd["timeCourse"].set_title('mean source activity', fontdict=font)
                axd["timeCourse"].legend(fontsize=12, loc='upper left')
                axd["timeCourse"].set_ylim([0,10])
                axd["timeCourse"].axvspan(tmin, tmax, facecolor='k', alpha = 0.1)
                
                if condi2 == 'GOu_GOc':
                    contra = 'GOu > GOc'
                elif condi2 == 'NoGo_GOu':
                    contra = 'NoGo > GOu'
                    
                axd["screenshot"].set_title(contra + ' at ' + time_pos2 + ': ' + hemi, fontdict=font)
                
                report.add_figure(fig, title= condi2 + '_' + time_pos2 +'_' +hemi+'_fROI_' + str(li) , replace = True)
                
                stc_fig1.close() 
                plt.close('all')    
          
# finally saving the report after the for subject loop ends.     
print('Saving the reports to disk')  
report.title = 'Group (n = ' + str(n_subs) + ') STCs perCondi ROIs from evkContrast Panel at_' + evnt+ ': '+ version + '_'  + method + '_' + ampliNormalization + '_' + scale 
extension = 'group_STCs_perCondi_ROIs_from_evkContrast_Panel'
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt +'_' + version+ '_' + method + '_' + ampliNormalization + '_' + scale +'_ppt.html', overwrite=True)            
          
   

            

               
            
    #%%                    
            # fig.tight_layout()
                      
        
        #     extension = 'stc_funclabel_evkContrast_' + condi2 + '_latView_' + time_pos2
        #     fname = op.join(config_for_gogait.eeg_dir_GOODremove,
        #                          config_for_gogait.base_fname_generic.format(**locals()))
        #     print("saving label: ", fname)
            
        #     mne.write_label(fname, func_label_LH, verbose=None)
                   
            
            
        
        # #%  MEDIAL VIEW        
        # stc_fig2 =  stc_mean_timepts.plot(
        #                     views=["med"],
        #                     hemi= "split",
        #                     smoothing_steps=7, 
        #                     size=(wt, ht),
        #                     view_layout = 'vertical',
        #                     time_viewer=False,
        #                     show_traces=False,
        #                     colorbar= False,
        #                     background='white',
        #                     clim = clim, # in values
        #                     brain_kwargs = dict(surf = 'inflated'),
                            
        #                 )  
        
        # # my_label= 'G_and_S_cingul-Mid-Post' #8 middle poste cingu gyrus and sulcus pMCC
        # # label_lh = [label for label in labels if label.name == my_label+"-lh"][0]
        # # label_rh = [label for label in labels if label.name == my_label+"-rh"][0]
        
        # ### Functional labels 
        # stc_with_zeros = stc_mean_timepts.copy()
        # data = np.abs(stc_with_zeros.data)
        # stc_with_zeros.data[data < vth_label] = 0.0

        # # 8.5% of original source space vertices were omitted during forward
        # # calculation, suppress the warning here with verbose='error'
        # func_labels_all_LH,  func_labels_all_RH = mne.stc_to_label(
        #     stc_with_zeros,
        #     src=src,
        #     smooth=True,
        #     # subjects_dir=subjects_dir,
        #     connected=True,
        #     verbose="error",
        # )
        
        # # take first as func_labels are ordered based on maximum values in stc
        # func_label_LH = func_labels_all_LH[0]
        # func_label_RH = func_labels_all_RH[0]
       
        # # show both labels
        # stc_fig2.add_label(func_label_LH, borders=True, color='k')
        # stc_fig2.add_label(func_label_RH, borders=True, color='k')
        
        # extension = 'stc_funclabel_evkContrast_' + condi2 + '_medView_' + time_pos2
        # fname = op.join(config_for_gogait.eeg_dir_GOODremove,
        #                      config_for_gogait.base_fname_generic.format(**locals()))
        # print("saving label: ", fname)
        
        # mne.write_label(fname, func_label_LH, verbose=None)
        # mne.write_label(fname, func_label_RH, verbose=None)
        
        # screenshot2 = stc_fig2.screenshot()

        # #%DORSAL VENTRAL VIEW     
        # ### https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_scalar_bar.html          
        # stc_fig3 =  stc_mean_timepts.plot(
        #                     views=["dor", "ven"],
        #                     hemi= "both",
        #                     smoothing_steps=7, 
        #                     size=(wt, ht),
        #                     view_layout = 'horizontal',
        #                     time_viewer= False,
        #                     show_traces= False,
        #                     colorbar= False,
        #                     background='white',
        #                     clim = clim, # in values
        #                     brain_kwargs = dict(surf = 'inflated'), 
                            
                                                                                 
        #                 )  
        
        # ### Functional labels 
        # stc_with_zeros = stc_mean_timepts.copy()
        # data = np.abs(stc_with_zeros.data)
        # stc_with_zeros.data[data < vth_label] = 0.0

        # # vth_label % of original source space vertices were omitted during forward
        # # calculation, suppress the warning here with verbose='error'
        # func_labels_all_LH,  func_labels_all_RH = mne.stc_to_label(
        #     stc_with_zeros,
        #     src=src,
        #     smooth=True,
        #     # subjects_dir=subjects_dir,
        #     connected=True,
        #     verbose="error",
        # )
        
        # # take first as func_labels are ordered based on maximum values in stc (more points)
        # func_label_LH = func_labels_all_LH[0]
        # func_label_RH = func_labels_all_RH[1]
       
        # # show both labels
        # stc_fig3.add_label(func_label_LH, borders=True, color='k')
        # stc_fig3.add_label(func_label_RH, borders=True, color='k')
        
        # extension = 'stc_funclabel_evkContrast_' + condi2 + '_venView_' + time_pos2
        # fname = op.join(config_for_gogait.eeg_dir_GOODremove,
        #                      config_for_gogait.base_fname_generic.format(**locals()))
        # print("saving label: ", fname)
        
        # mne.write_label(fname, func_label_LH, verbose=None)
        # mne.write_label(fname, func_label_RH, verbose=None)
        
               
        # screenshot3 = stc_fig3.screenshot()
       
        # ax_ind = 0
        # for ax, image in zip([axes[figInd + ti2],axes[figInd+ nT+ ti2],axes[figInd + 2*nT + ti2]],
        #                      [screenshot1, screenshot2, screenshot3]):
        #     # ax.set_xticks([])
        #     # ax.set_yticks([])
        #     ax.axis('off')
        #     ax.imshow(image)
        #     ax_ind = ax_ind + 1
        
        # fig.tight_layout()
        
        # # stc_fig1.close() 
        # # stc_fig2.close()
        # # stc_fig3.close()
        
            
#     
           


    
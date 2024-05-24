#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
Single sub analysis: Source reconstruction using template MRI
15a.Generate STC and save the dipoles (vector format) as numpy. Report the foward and noise cov matrix

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
from mne.datasets import fetch_fsaverage
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse 
from mne.minimum_norm import write_inverse_operator
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator

import config_for_gogait


version_list = ['CHANremove'] # 'GOODremove' | 'CHANremove'

method = "dSPM"  # "MNE" | "dSPM"

condi_name = ['GOc', 'GOu', 'NoGo']

event_type = ['target']   # 'target' | 'cue'

orientation = 'varyDipole'  # 'varyDipole' | 'fixed'

for subject in config_for_gogait.subjects_list: 
    for veri, version in enumerate(version_list):
        eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
        print("Processing subject: %s" % subject)
        # just for cov estimation 
        condi_name_all = ['GOc', 'GOu', 'NoGo']
        event_type_for_cov = ['cue']
        # epochs_array_for_cov = np.ones([1,1,1])*np.nan # create empty array 
        """ step 0: concatenate epochs of different conditions at CUE ONLY to compute noise cov matrix"""  
        for ci, condi in enumerate(condi_name_all):
            for ei, evnt in enumerate(event_type_for_cov):
                print('  importing epochs numpy array from disk')
                extension = condi_name_all[ci] +'_' + event_type[ei] + '_' + version + '_epo'
                epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                          config_for_gogait.base_fname.format(**locals()))
                print("Input: ", epochs_fname)
                epochs = mne.read_epochs(epochs_fname)
                # check: epochs.info['bads'] 
                ep_data = epochs.get_data()
                if ci == 0:
                    epochs_array_for_cov = ep_data
                else:
                    epochs_array_for_cov = np.vstack((epochs_array_for_cov, ep_data))
        
        epochs_for_cov = mne.EpochsArray(epochs_array_for_cov, info = epochs.info, 
                                         tmin = -0.2, baseline = (None,0))
                    
                
                
    #%%
        """ step 1: import epochs of different conditions and ISI"""    
        
                
        # create mne reports for saving plots 
        report = mne.Report() 
        #% for loop of conditions with ISIs
        
        
        
        for ci, condi in enumerate(condi_name):
            for ei, evnt in enumerate(event_type):
                print('  importing the epochs numpy array from disk')
                extension = condi_name[ci] +'_' + event_type[ei] + '_' + version + '_epo'
                epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                          config_for_gogait.base_fname.format(**locals()))
                print("Input: ", epochs_fname)
                epochs = mne.read_epochs(epochs_fname)
                epochs = epochs.set_eeg_reference(projection=True)
                info = epochs.info ### NOTE: simply not copying, but assigning bidirectionally 
                
                # check: epochs.info['bads'] 
                
                
                ## OPTION 1: actiCAP128-56.pos OR actiCAP128-60.pos
                # loading the channel position: from Laurent
                # actiPOS = '/network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait/EEG/actiCAP128-60.pos'
                # df_ch_pos = pd.read_table(actiPOS)
                # df_ch_pos.columns = ['ch_names', 'X', 'Y', 'Z']
                # chname = df_ch_pos['ch_names'].to_list()
                # chpos = df_ch_pos[["X", "Y", "Z"]].apply(lambda x: np.array([x["X"], x["Y"], x["Z"]]), axis=1).to_numpy()
                # chpos = chpos/100 # location units conversion from cm to m 
                # ch_pos_dict = dict(zip(chname,chpos))  # channel position                
                # montage = mne.channels.make_dig_montage(ch_pos_dict, coord_frame = 'head')
                # # montage.plot(kind = '3d')
                # info.set_montage(montage, on_missing = 'warn')
                
        ## -------------------------------------------------------------------------------------------------------       
                ## OPTION 2: standard_1005 ("works the best") OR brainproducts-RNP-BA-128
                # # Read and set the EEG electrode locations, which are already in fsaverage's
                # # space (MNI space) for standard_1020: 
                # #    standard_1005, brainproducts-RNP-BA-128
                montage_std = mne.channels.make_standard_montage("standard_1005") 
                mne.rename_channels(info, mapping = {'O9':'I1', 'O10':'I2'}, allow_duplicates=False)
                info.set_montage(montage_std, on_missing = 'warn') 
                # montage_std.plot(kind = '3d')
                   
         ## -------------------------------------------------------------------------------------------------------              
                #  ## OPTION 3: actiCAP128 generic or passiCAP(brainCAP)
                # ## Read the actiCAP file
                # fname_bv = '/network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait/montage/active/actiCAP128_Channel/AC-128.bvef'
                # montage_acti = mne.channels.read_custom_montage(fname_bv)
                # info.set_montage(montage_acti, on_missing = 'warn')
                # # ch_pos = np.array(list(montage.get_positions()['ch_pos'].values()))
                
                # ## Read the passiCAP file
                # fname_bv = '/network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait/montage/passive/BrainCap128_Channel/BC-128.bvef'
                # montage_passi = mne.channels.read_custom_montage(fname_bv)
                # montage_passi.plot(kind = '3d')
                # info.set_montage(montage_passi, on_missing = 'warn')
        ## -------------------------------------------------------------------------------------------------------              
               #  ## OPTION 4: actiCAP128 positions obtained at ICM polhemus system
                
                # ## step 1: import the fif file and remove the reference (redundant) electrode position 
                # fif_file = '/network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait/EEG/acticap_uol_58_128.fif'
                # montage_fif = mne.channels.read_dig_fif(fif_file)
                # montage_fif.dig # print to see the naming of ref channel to be removed
                # montage_fif.dig.pop(7) # remove 'EEG #0' from ch pos list
                # montage_fif.ch_names.remove('EEG000') # correspondingly remove from ch name list
                
                # #montage_fif.plot(sphere = 0.09, show_names=True) # now plot and see
                # # evoked[0].plot_sensors(ch_type="eeg", sphere = 0.4, show_names=True)
                
                # # step 2: get the ch positions
                # ch_pos_dict = montage_fif._get_ch_pos() # in dict format
                # ch_pos_array = np.array(list(montage_fif.get_positions()['ch_pos'].values()))
                # ch_names = montage_fif._get_dig_names()
                # nas = ch_pos_array = np.array(list(montage_fif.get_positions()['nasion'])).flatten()
                # lpa = np.array(list(montage_fif.get_positions()['lpa'])).flatten()
                # rpa = np.array(list(montage_fif.get_positions()['rpa'])).flatten()
                # hpi = montage_fif.get_positions()['hpi']
                
                # # step 3: copy the ch names and pos from array into excel sheet for renaming. 
                # # step 4: import the renamed chs and pos from .csv file.
                # df = pd.read_csv('ch_pos_renamed_uol58_128.csv')
                # chname = df['new'].tolist()           
                # chpos = df[['pos1', 'pos2', 'pos3']].to_numpy() # already in SI unit (m)
                # ch_pos_dict = dict(zip(chname,chpos))  # channel position                
                
                # # step 5: make the montage by adding all info
                # montage_icm = mne.channels.make_dig_montage(ch_pos_dict)
                
              
                
                # # montage_icm.plot(kind = '3d')
                # # mne.rename_channels(info, mapping = {'POz':'POz'}, allow_duplicates=False)
                # info.set_montage(montage_icm, on_missing = 'warn')
                
                
                
    
                #%% Source imaging using template MRI
                
                if ci==0 and ei==0:  
                    ## STEP 1: FORWARD COMPUTATION (only once: same across condi and event)
                    ## using the standard template MRI subject, 'fsaverage'
                    ## Adult template MRI (fsaverage)
                    
                    # Download fsaverage files
                    fs_dir = fetch_fsaverage(verbose=True)
                    template_subjects_dir = op.dirname(fs_dir)
                    
                    # The files live in:
                    template_subject = "fsaverage"
                    
                    ## 1a: Co-registration of landmarks of head with template MRI 
                    """ [1.TRANS matrix] """
                    ### define trasformation matrix 
                    trans = op.join(fs_dir, "bem", "fsaverage-trans.fif")
                    
                    plot_kwargs = dict(
                                    subject= template_subject,
                                    subjects_dir= template_subjects_dir,
                                    # surfaces="head-dense",
                                    dig= "fiducials",
                                    # meg= True,
                                    eeg=["original", "projected"],
                                    show_axes=True,
                                    # coord_frame= 'auto',
                                )
                    # fiducials = "estimated"  # get fiducials from fsaverage
                    # coreg = Coregistration(info, template_subject, template_subjects_dir, fiducials=fiducials)
                    # trans = coreg.trans
                   
                    """## 31/01/2024 :  NO COREG and Auto correction is required now"""
                    ## we are taking MNE's std trans 
                    ## https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html#sphx-glr-auto-tutorials-forward-35-eeg-no-mri-py
                    ## Read and set the EEG electrode locations, which are already in fsaverage's
                    ## space (MNI space) for standard_1020:
                    
                    # trans = "fsaverage"  # MNE has a built-in fsaverage transformation
                    # fig_before = mne.viz.plot_alignment(info, trans = trans, mri_fiducials=True, **plot_kwargs)
                    
                    
                    #%% Automatic correction of co-registration 
                    
                    # # coreg.fit_fiducials(verbose=True)
                    # # fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
                    # # coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
                    # # fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)  
                    # coreg.omit_head_shape_points(distance=0.5 / 1000)  # distance is in meters
                    # # coreg.fit_icp(n_iterations=50, nasion_weight=10.0, verbose=True)
                    # coreg.fit_icp(n_iterations=50, nasion_weight=10.0, verbose=True)
                    # fig_after = mne.viz.plot_alignment(info, trans = coreg.trans, mri_fiducials=True, **plot_kwargs)
                    
                    # # #"fsaverage"  # MNE has a built-in fsaverage transformation
                    
                    # # # 1b: define electrode position  
                    # ## info = epochs.info.copy()
                    
                   
        #%%            
                    """ [2. SRC file] """
                    # 1c: define source space 
                    src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
                    # src_vol = op.join(fs_dir, "bem",'fsaverage-vol-5-src.fif')
                    
                    # Check that the locations of EEG electrodes is correct with respect to MRI
                    # fig_coreg = mne.viz.plot_alignment(
                    #     subject = template_subject,
                    #     info = info,
                    #     src=src,
                    #     eeg=["original","projected"],
                    #     trans= trans,
                                          
                    #     coord_frame="head",
                    #     subjects_dir= template_subjects_dir
                    # )
                    
                    # report.add_figure(fig_coreg, title = 'template_subject_co-registration', replace = True)
                    report.add_trans(trans = trans,
                                    subject = template_subject,
                                    subjects_dir = template_subjects_dir,
                                    info = info,
                                    title="template_sub_COREG")
                    plt.close()
      
                    #%% 1d: importing and plotting bem file of fsaverage subject [BEM]
                    """ [3. BEM solution] """
                    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
                    plot_bem_kwargs = dict(
                        subject = template_subject,
                        subjects_dir = template_subjects_dir,
                        brain_surfaces="white",
                        orientation="coronal",
                        slices=[50, 100, 150, 200]
                    )
                    
                    fig_bem = mne.viz.plot_bem(**plot_bem_kwargs)
                    report.add_figure(fig_bem, title = 'template_sub_BEM', replace = True)
                    
                    report.add_bem(subject = template_subject, 
                                   subjects_dir= template_subjects_dir, 
                                   title="tempMRI & tempBEM", decim=20, width=256)
                    
                    #%% 1e: compute foward solution 
                    """ [4. FWD solution]"""
                    #### N.B. USING different info due to error #####
                    #### RuntimeError: Missing EEG channel location. 
                    
                    fwd = mne.make_forward_solution(info = info, trans = trans, 
                                                    src = src, bem=bem, eeg=True,
                                                    mindist=5.0, n_jobs=None)
                    
                    report.add_forward(forward = fwd, title="template_subj_FWD")
                    
                    # print(fwd)                
                    ## Notes: This function is designed to provide
                    ## All modern (Freesurfer 6) fsaverage subject files
                    ## All MNE fsaverage parcellations
                    ## fsaverage head surface, fiducials, head<->MRI trans, 1- and 3-layer BEMs (and surfaces)
                    
                
                #%% STEP 2: Compute regularized noise covariance per subject (same for all condi)
                    """ [5. COV matrix] """
                    """ estimate the cov only for once (per sub) for all condi at CUE and use the same for target"""
                
               
                    noise_cov = mne.compute_covariance(epochs_for_cov, tmin = -0.2, tmax=0.0,
                                                       method=["shrunk", "empirical"], 
                                                       rank=None, verbose=True)
                    
                
                    # fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info = info)
                    report.add_covariance(cov = noise_cov, info = epochs_for_cov.info,
                                          title= 'GOc_GOu_NoGO_cue_COV')
                    
                    print("Writing noise covariance matrix as FIF to disk")
                    extension = version +'_cov' # keep _inv in the end
                    fname_out = op.join(eeg_subject_dir_GOODremove,
                                         config_for_gogait.base_fname.format(**locals()))
                    
                    print("Output: ", fname_out)
                    mne.write_cov(fname_out, noise_cov, overwrite = True, verbose = None)
                    
                
                
                    #%% STEP 3: compute the inverse per subject (same for all condi)
                    
                    """loose = float | ‘auto’ | dict"""
                    ## Value that weights the source variances of the dipole components that are parallel (tangential) 
                    ## to the cortical surface. Can be:
                    ## float between 0 and 1 (inclusive). If 0, then the solution is computed with fixed orientation. 
                    ## """If 1, it corresponds to free orientations.""" 
                    ## 'auto' (default) Uses 0.2 for surface source spaces (unless fixed is True) and 
                    ## 1.0 for other source spaces (volume or mixed). 
        
                    """ fixed = bool | ‘auto’"""
                    ## Use fixed source orientations normal to the cortical mantle. 
                    ## If True, the loose parameter must be “auto” or 0. If ‘auto’, the loose value is used. 
                    
                    ## synthesis: if fixed == 'True',loose == 0, and orient = None,  signed activation (+/-)
                    ##            if fixed == 'True', loose == 1, and orient = None , abs activation (+ve only)
                    ##            if fixed == 'False', and loose == 0, same as case 1  if orient = None
                    ##            if fixed == 'False', and loose == 1, same as case 2 with var % and amp changes 
                    
                   ##  if fixed == 'False', and loose == 0.2, same as case 1  if orient = "vector"
                    
                    # looseVal = 1 # was set 0.2 before
                    
                   
                    if orientation == 'varyDipole':
                        looseVal = 1
                    elif orientation == 'fixDipole':
                        looseVal = 0.2
                        
                    
                    inverse_operator = make_inverse_operator(info, fwd, noise_cov, fixed = 'False',
                                                             loose = looseVal, depth=0.8)
                    report.add_inverse_operator(inverse_operator = inverse_operator, 
                                               title= 'GOc_GOu_NoGO_cue_INV')
                    
                    print("Writing inverse operator as FIF to disk")
                    extension = version +'_'+ orientation +'_'+'inv' # keep _inv in the end
                    fname_out = op.join(eeg_subject_dir_GOODremove,
                                         config_for_gogait.base_fname.format(**locals()))
                    
                    print("Output: ", fname_out)
                    write_inverse_operator(fname_out, inverse_operator, overwrite = True, verbose = None)               
                    
                
                # del fwd  
                #%% STEP 4: Compute inverse solution per condi and per sub
                
                  # “MNE” | “dSPM” | “sLORETA” | “eLORETA”
                snr = 3.0
                lambda2 = 1.0 / snr**2
                
                ## ValueError: EEG average reference (using a projector) is mandatory for modeling,
                ## use the method set_eeg_reference(projection=True)
                
                """ pick_ori = None | “normal” | “vector” """
                """ "None"- Pooling is performed by taking the norm of loose/free orientations."""
                ##  In case of a fixed source space no norm is computed leading to signed source activity. 
                ## "normal"- Only the normal to the cortical surface is kept. 
                ##  This is only implemented when working with loose orientations. 
                ## "vector" - No pooling of the orientations is done, 
                ## and the vector result will be returned in the form of a mne.VectorSourceEstimate object. 
                
                # updated on 23/02/2024, this eef_ref_proj is placed above in info, to pass this info to inv operator
                evoked = epochs.average().pick("eeg")
                stc, residual = apply_inverse(
                                            evoked,
                                            inverse_operator,
                                            lambda2,
                                            method=method,
                                            pick_ori = "vector",
                                            return_residual = True,
                                            verbose=True)
                ### bug when using "vector": Stays in infinite loop when saving. overwriting existing file. 
                ### SOLUTION: saving STC files as numpy format 
                
                print('  Writing the stc to disk')
               
                extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
                stc_fname_array = op.join(eeg_subject_dir_GOODremove,
                                      config_for_gogait.base_fname_npy.format(**locals()))
                print("Output: ", stc_fname_array)
                stc_data = stc.data
                np.save(stc_fname_array, stc_data)
                
                #%% 
                
                plt.close('all')
         
     
        ### finally saving the report after the for loop ends.     
        print('Saving the reports to disk')   
        report.title = 'Forward_model_and_noise_cov: ' + subject + '_at_'+ evnt +'_' + version +'_' + orientation + '_' + method 
        report_fname = op.join(config_for_gogait.report_dir_GOODremove,subject)
        report.save(report_fname+'_fwd_std1005_noisecov_cue' +'_at_'+ evnt +'_' + version +'_' + orientation + '_' + method  + '.html', overwrite=True)

    
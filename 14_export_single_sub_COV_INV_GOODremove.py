#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:17:41 2023

@author: harish.gunasekaran

"""

"""
=============================================
Single sub analysis: computing cov matrix and inv operator per sub and saving to the disk. 
https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html#sphx-glr-auto-tutorials-forward-35-eeg-no-mri-py
https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

==============================================


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

# version = 'GOODremove' OR 'CHANremove'
# version_list = ['GOODremove','CHANremove']

version_list = ['CHANremove']

for sb, subject in enumerate(config_for_gogait.subjects_list): 
    
    for veri, version in enumerate(version_list):
    
        eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
        print("Processing subject: %s" % subject)
        
        # create mne reports for saving plots 
        report = mne.Report(title = subject)
            
        condi_name = ['GOc', 'GOu', 'NoGo']
        event_type = ['cue', 'target']
        event_type_for_cov = ['cue']
        # epochs_array_for_cov = np.ones([1,1,1])*np.nan # create empty array 
        """ step 0: concatenate epochs of different conditions at CUE ONLY to compute noise cov matrix"""  
        for ci, condi in enumerate(condi_name):
            for ei, evnt in enumerate(event_type_for_cov):
                print('  importing epochs numpy array from disk')
                extension = condi_name[ci] +'_' + event_type[ei] + '_' + version + '_epo'
                epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                          config_for_gogait.base_fname.format(**locals()))
                print("Input: ", epochs_fname)
                epochs = mne.read_epochs(epochs_fname)
                ep_data = epochs.get_data()
                if ci == 0:
                    epochs_array_for_cov = ep_data
                else:
                    epochs_array_for_cov = np.vstack((epochs_array_for_cov, ep_data))
        
        epochs_for_cov = mne.EpochsArray(epochs_array_for_cov, info = epochs.info, 
                                         tmin = -0.2, baseline = (None,0))
        
        if sb == 0: # do the co-registration and fwd computation only once and is fixed for all sub. 
        
            info = epochs.info ### NOTE: simply not copying, but assigning bidirectionally 
            ## OPTION 2: standard_1005 ("works the best") OR brainproducts-RNP-BA-128
            # # # Read and set the EEG electrode locations, which are already in fsaverage's
            # # # space (MNI space) for standard_1020: 
                # standard_1005, brainproducts-RNP-BA-128
            montage_std = mne.channels.make_standard_montage("standard_1005") 
            mne.rename_channels(info, mapping = {'O9':'I1', 'O10':'I2'}, allow_duplicates=False)
            info.set_montage(montage_std, on_missing = 'warn') 
            
            ## STEP 1: FORWARD COMPUTATION (only once)
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
            # trans = op.join(fs_dir, "bem", "fsaverage-trans.fif")
            
            plot_kwargs = dict(
                            subject= template_subject,
                            subjects_dir= template_subjects_dir,
                            surfaces="head-dense",
                            dig="fiducials",
                            meg=[],
                            eeg=['original'],
                            show_axes=True,
                            coord_frame= 'auto',
                        )
            fiducials = "estimated"  # get fiducials from fsaverage
            coreg = Coregistration(info, template_subject, template_subjects_dir, fiducials=fiducials)
            
            #%% Automatic correction of co-registration 
            
            # fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)  
            coreg.omit_head_shape_points(distance=0.5 / 1000)  # distance is in meters
            coreg.fit_icp(n_iterations=50, nasion_weight=10.0, verbose=True)
            # fig_after = mne.viz.plot_alignment(info, trans = coreg.trans, mri_fiducials=True, **plot_kwargs)
            
            #%% """ [2. SRC file] """
            # 1c: define source space 
            src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
            
            #%%""" [3. BEM solution] """
            bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
            # plot_bem_kwargs = dict(
            #     subject = template_subject,
            #     subjects_dir = template_subjects_dir,
            #     brain_surfaces="white",
            #     orientation="coronal",
            #     slices=[50, 100, 150, 200]
            # )
            
            # fig_bem = mne.viz.plot_bem(**plot_bem_kwargs)
            
            #%%  """ [4. FWD solution]"""
              #### N.B. USING different info due to error #####
              #### RuntimeError: Missing EEG channel location. 
            
            fwd = mne.make_forward_solution(info = info, trans = coreg.trans, 
                                            src = src, bem=bem, eeg=True,
                                            mindist=5.0, n_jobs=None)
        
        #%%""" [5. COV matrix] """ CHANGES PER SUBJECT
        """ estimate the cov only for once (per sub) for all condi at CUE and use the same for target"""
        
        
        noise_cov = mne.compute_covariance(epochs_for_cov, tmin = -0.2, tmax=0.0,
                                           method=["shrunk", "empirical"], 
                                           rank=None, verbose=True)    
        
        print("Writing noise covariance matrix as FIF to disk")
        extension = version +'_'+'cov' # keep _cov in the end
        fname_out = op.join(eeg_subject_dir_GOODremove,
                             config_for_gogait.base_fname.format(**locals()))
        
        print("Output: ", fname_out)
        mne.write_cov(fname_out, noise_cov, overwrite = True, verbose = None)
            
         #%% [6. compute the inverse] CHANGES PER SUBJECT
         
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
         
        looseVal = 1 # was set 0.2 before
        
        if looseVal == 1:
            orientation = 'varyDipole'
        else:
            orientation = 'fixDipole'
            
        inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                                 fixed = 'False',loose = looseVal, depth=0.8) 
        
        print("Writing inverse operator as FIF to disk")
        
        extension = version +'_'+ orientation +'_'+'inv' # keep _inv in the end
        fname_out = op.join(eeg_subject_dir_GOODremove,
                             config_for_gogait.base_fname.format(**locals()))
        
        print("Output: ", fname_out)
        write_inverse_operator(fname_out, inverse_operator, overwrite = True, verbose = None)               

     
    
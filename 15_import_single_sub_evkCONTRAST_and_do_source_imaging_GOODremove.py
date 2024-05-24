#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:02:43 2023

@author: harish.gunasekaran
"""
"""
=============================================
Single sub analysis: Source reconstruction of EVOKED CONTRAST using template MRI
15a.Generate STC and save the dipoles (vector format) as numpy. Report the foward and noise cov matrix

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
from mne.minimum_norm import read_inverse_operator
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator

import config_for_gogait

# version = 'GOODremove' OR 'CHANremove'
# version_list = ['GOODremove','CHANremove']
# ampliNormalization = ['AmpliNorm', 'AmpliActual']

ampliNormalization = 'AmpliActual'

version_list = ['CHANremove']

event_type = ['target']

orientation = 'varyDipole' #  ['fixDipole', 'varyDipole' ]

method = "dSPM"  # "MNE" | "dSPM"

contrast_kind = ['GOu_GOc','NoGo_GOu']   


for veri, version in enumerate(version_list):
    
    for ei, evnt in enumerate(event_type):
    
        for contra, contrast in enumerate(contrast_kind): # two contrasts
                       
            for subject in config_for_gogait.subjects_list: 
                
                eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                print("Processing subject: %s" % subject)        
                                
                #%% reading the contrast evoked from disk
                print('reading the evoked contrast from disk')
                extension =  contrast +'_' + evnt +'_' + version +'_' + ampliNormalization +'_ave'
                evoked_fname = op.join(eeg_subject_dir_GOODremove,
                                        config_for_gogait.base_fname.format(**locals()))
                print("Input: ", evoked_fname)
                evoked = mne.read_evokeds(evoked_fname)
                evkd = evoked[0].pick('eeg').set_eeg_reference(ref_channels='average', projection=True)
                #%% reading inverse operator per sub
                print("Reading inverse operator as FIF from disk")
               
                extension = version +'_'+ orientation +'_'+'inv' # keep _inv in the end
                    
                fname_in = op.join(eeg_subject_dir_GOODremove,
                                     config_for_gogait.base_fname.format(**locals()))
                
                print("input: ", fname_in)
                inverse_operator = read_inverse_operator(fname_in)                  
                
                #%% apply inverse method to evoked contrast 
                # method = "MNE"  # “MNE” | “dSPM” | “sLORETA” | “eLORETA”
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
                
                stc, residual = apply_inverse(
                                            evkd,
                                            inverse_operator,
                                            lambda2,
                                            method=method,
                                            pick_ori = "vector",
                                            return_residual = True,
                                            verbose=True)
                ### bug when using "vector": Stays in infinite loop when saving. overwriting existing file. 
                ### SOLUTION: saving STC files as numpy format 
              #%%  
                print('  Writing the stc to disk')
               
                extension = contrast +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
                stc_fname_array = op.join(eeg_subject_dir_GOODremove,
                                     config_for_gogait.base_fname_npy.format(**locals()))
                print("Output: ", stc_fname_array)
                stc_data = stc.data
                np.save(stc_fname_array, stc_data)
                
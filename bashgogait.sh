#!/bin/bash
#SBATCH --job-name=gogait_source
#SBATCH --partition=normal
#SBATCH --time=100:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait/scripts
#SBATCH --output=file_output_%j.log
#SBATCH --error=error_output_%j.log

module load MNE/1.3.0
mne
ls -l
which python
python
python < 55c_import_group_sub_TF_make_groupTF_plots_topoplots_GOODpatients.py
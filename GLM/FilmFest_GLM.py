import os
import glob
from nilearn.image import mean_img
from nilearn.plotting import plot_anat, plot_img, plot_stat_map, show
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn.plotting import plot_contrast_matrix
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import sys

#/Users/rohinpalsule/Desktop/GLM/
class SequentialityGLM:
    """
    Pipeline and helper functions for getting PM Sequenitality across Filmfest
    """
    def __init__(self,subID:str="01", bold_path:str=f"/mnt/data1/bornstein/"):
        self.SUB_ID = subID # I'm sorry I always forget what version of subid subID SubID SUB_ID sub_id sub-id subject-id id ID I should be using
        self.TR = 1.5
        self.func_filename_1 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.SUB_ID}','func',f'sub-{self.SUB_ID}_task-movie_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        self.func_filename_2 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.SUB_ID}','func',f'sub-{self.SUB_ID}_task-movie_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        self.func_scans = [self.func_filename_1,self.func_filename_2]
        self.brain_mask = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.SUB_ID}','anat',f'sub-{self.SUB_ID}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
        
        # Need to deside what we want to use as confounds
        self.confound_filename_1 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.SUB_ID}','func',f'sub-{self.SUB_ID}_task-movie_run-01_desc-confounds_timeseries.tsv')
        self.confound_filename_2 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.SUB_ID}','func',f'sub-{self.SUB_ID}_task-movie_run-02_desc-confounds_timeseries.tsv')

        # Events filenames
        self.event_filename_1 = os.path.join('../sequentiality_events',f'filmfest_sequentiality_run-01_events.tsv')
        self.event_filename_2 = os.path.join('../sequentiality_events',f'filmfest_sequentiality_run-02_events.tsv')

        # Creates a path for Localzier outputs
        self.output_dir = Path.cwd() / "results" / "GLM"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.subject_cache_dir = Path.cwd() / "nilearn_cache"
        self.subject_cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output will be saved to: {self.output_dir}")
        print(f"Running GLM on participant {self.SUB_ID}")

    def define_first_level(self):
        """
        Define GLM Model, our scanner uses these for everything so I have left it fixed but adjust TR if needed
        """
        fmri_glm = FirstLevelModel(
            t_r=self.TR,
            smoothing_fwhm=6,
            noise_model="ar1",
            standardize=False,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=1/128,
            verbose=1,
            memory=str(self.subject_cache_dir),   # or Memory(location=subject_cache_dir) directly
            memory_level=1,
        )
        return fmri_glm

    def make_confounds_df(self):

        # Loads all confound tsv files
        confound_files = [self.confound_filename_1,self.confound_filename_2]

        confound_dfs = []
        motion_cols = [
            "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
            "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
            "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
            "trans_x_power2", "trans_y_power2", "trans_z_power2",
            "rot_x_power2", "rot_y_power2", "rot_z_power2",
            "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
            "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
        ]  # full 24-parameter motion model
        wm_cols = ["white_matter", "white_matter_derivative1"]
        csf_cols = ["csf", "csf_derivative1"]
        gsr_cols = ["global_signal", "global_signal_derivative1"]

        selected_cols = motion_cols + wm_cols + csf_cols
        for c_file in confound_files:
            confound_df = pd.read_csv(c_file,sep='\t')[selected_cols]
            confound_df = confound_df.fillna(0)
            print(len(confound_df))
            confound_dfs.append(confound_df)

        return confound_dfs

    def fit_glm(self,fmri_glm):
        concat_event_dfs = []
        for event_path in [self.event_filename_1,self.event_filename_2]:
            concat_event_dfs.append(pd.read_csv(event_path,sep='\t'))
        confound_dfs = self.make_confounds_df()
        fit_glm = fmri_glm.fit(run_imgs=self.func_scans, events=concat_event_dfs,confounds=confound_dfs)

        return fit_glm

    
    def get_contrast_nii(self, fmri_glm, pm_column: str = 'Movie', weight: float = 1):

        z_map = fmri_glm.compute_contrast(pm_column, output_type='effect_size')

        return z_map

    def run_glm(self, map_name:str='Sequentiality_PM_z_map'):
        map_name = map_name + f'_sub-{self.SUB_ID}'
        glm = self.define_first_level()
        fit_glm = self.fit_glm(glm)
        z_map = self.get_contrast_nii(fit_glm)
            
        output_path = os.path.join(self.output_dir,"contrast_maps")

        z_map.to_filename(os.path.join(output_path,f"{map_name}.nii.gz"))

if __name__ == '__main__':
    sub_id = sys.argv[1]
    pm_glm = SequentialityGLM(subID=sub_id)
    pm_glm.run_glm()
    print('All done!')


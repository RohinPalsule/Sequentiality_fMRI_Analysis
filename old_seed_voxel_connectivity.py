import os
import sys
import glob
from nilearn import datasets, plotting, image
from nilearn.image import concat_imgs
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiSpheresMasker
from nilearn.maskers import NiftiMasker
from nilearn.maskers import NiftiLabelsMasker
from pathlib import Path
import pandas as pd
import hashlib
from nilearn.image import index_img, load_img
import csv


class SeedConnectivity():
    def __init__(self,sub_id:str='01',bold_path:str='/mnt/data1/bornstein/'):
        self.ID = sub_id
        self.TR = 1.5
        self.func_filename_1 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.ID}','func',f'sub-{self.ID}_task-movie_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
        self.func_filename_2 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.ID}','func',f'sub-{self.ID}_task-movie_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')

	# Brain mask
        self.brain_mask = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.ID}','anat',f'sub-{self.ID}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')

        # Need to deside what we want to use as confounds
        self.confound_filename_1 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.ID}','func',f'sub-{self.ID}_task-movie_run-01_desc-confounds_timeseries.tsv')
        self.confound_filename_2 = os.path.join(bold_path,'LeeChenfMRIPrep',f'sub-{self.ID}','func',f'sub-{self.ID}_task-movie_run-02_desc-confounds_timeseries.tsv')

        # Events filenames
        self.event_filename_1 = os.path.join(bold_path,'LeeChenRaw','ds004042',f'task-movie_run-01_events.tsv')
        self.event_filename_2 = os.path.join(bold_path,'LeeChenRaw','ds004042',f'task-movie_run-02_events.tsv')

        # THIS IS THE PCC FROM THE TUTORIAL, NEED TO CHANGE
        self.seed_sphere_coords = [(0,-53,26)] # This is the PCC seed used in Brandman et al. 2021 :)

        self.output_dir = Path.cwd() / "results" / "plot_seed_to_voxel_correlation"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Output will be saved to: {self.output_dir}")
        print(f"Running seed for PCC coords: {self.seed_sphere_coords}. Make sure coordinates are correct!")

    def stable_hash(self,s):
        """For the ID to set the seed (going to need to remove soon!)"""
        return int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % (2**32)

    def get_block_bounds(self,events_path):
        """
        Reads appropriate event file and grabs the TRs corresponding to selected movie
        """
        events = pd.read_csv(events_path, sep="\t")
        blocks = []
        for _, row in events.iterrows():
            start_tr = int(row["onset"] / self.TR)
            end_tr = int((row["onset"] + row["duration"]) / self.TR)
            blocks.append((start_tr, end_tr))
        return blocks

    def select_block(self,func_path, confounds_path, block_bounds, block_idx):
        """
        Clips nifti and confounds that fall in the movie we are using for this
        """
        func_img = load_img(func_path)
        confounds_df = pd.read_csv(confounds_path, sep="\t")

        start_tr, end_tr = block_bounds[block_idx]
        block_func = index_img(func_img, slice(start_tr, end_tr))
        block_confounds = confounds_df.iloc[start_tr:end_tr].reset_index(drop=True)

        assert block_func.shape[-1] == len(block_confounds), \
            "func/confound TR mismatch - check onset/duration units against TR"

        return block_func, block_confounds

    def log_localizer_choice(self, subject_id, run, block_idx, block_bounds, log_path="localizer_assignments.csv"):
        log_path = Path(log_path)
        write_header = not log_path.exists()
        start_tr, end_tr = block_bounds[block_idx]
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["subject_id", "run", "block_idx", "start_tr", "end_tr"])
            writer.writerow([subject_id, run, block_idx, start_tr, end_tr])

    def choose_localizer_run_and_block(self,rng_seed=42):
        """
        Takes in scans and picks one random movie to clip Nifti and confounds.tsv file

        FIX: need to change sampling to ensure we have equal distribution across movies
        """
        rng = np.random.default_rng(rng_seed + self.stable_hash(self.ID))

        # Create dictionaries for each file type
        run_paths = {1:self.func_filename_1,
                        2:self.func_filename_2}
        confound_paths = {1:self.confound_filename_1,
                        2:self.confound_filename_2}
        event_paths = {1:self.event_filename_1,
                        2:self.event_filename_2}

        # Pick run 1 or 2 (need to do uniform sampling across participants in future)
        chosen_run = rng.choice(list(run_paths.keys()))  # 50/50 between run 1 and run 2

        # Pick random block idx in chosen run above
        block_bounds = self.get_block_bounds(event_paths[chosen_run])
        chosen_block_idx = rng.integers(0, len(block_bounds))

        # Grab paths for functional / confounds
        confounds_path = confound_paths[chosen_run]
        func_path = run_paths[chosen_run]

        # Clip scan and confound to be in the movie we want
        block_func, block_confounds = self.select_block(
            func_path, confounds_path, block_bounds, chosen_block_idx
        )

        # Logging which story each participant used for this (CHECK)
        self.log_localizer_choice(self.ID,chosen_run,chosen_block_idx,block_bounds)

        return chosen_run, chosen_block_idx, block_func, block_confounds

    def get_sphere_seed(self):
        seed_masker = NiftiSpheresMasker(
            self.seed_sphere_coords,
            radius=5.3,# gives us 78 voxels (paper does 80 voxels)
            smoothing_fwhm=6,
            detrend=True,
            standardize_confounds=True,
            standardize=True, # The original tutorial did not have this but this is how I got it mean centered and have the correlations interpretable 
            low_pass=0.1,
            high_pass=0.009,
            t_r=self.TR,
            memory="nilearn_cache",
            memory_level=1,
            verbose=1,
        )
        return seed_masker
    
    def fit_seed(self,seed_masker,func_file,confound_file):

        seed_time_series = seed_masker.fit_transform(
            func_file
        )
        return seed_time_series
    
    def mask_timeseries(self):
        """
        Need to adjust all hyperparameters
        """
        brain_masker = NiftiMasker(
            mask_img=self.brain_mask,
            smoothing_fwhm=6,
            detrend=True,
            standardize_confounds=True,
            standardize=True, # The original tutorial did not have this but this is how I got it mean centered and have the correlations interpretable 
            low_pass=0.1,
            high_pass=0.009,
            t_r=self.TR,
            memory="nilearn_cache",
            memory_level=1,
            verbose=1,
        )
        return brain_masker
    
    def fit_timeseries(self,brain_masker,func_file,confound_file):

        brain_time_series = brain_masker.fit_transform(
            func_file
        )

        return brain_time_series
    
    def plot_seed_timeseries(self,seed_time_series,region:str='Posterior Cingulate Cortex'):
        """
        Need to grab seed_time_series from self.fit_seed(), helper function for sanity checks
        """
        plt.figure(constrained_layout=True)
        plt.plot(seed_time_series)
        plt.title(f"Seed time series ({region})")
        plt.xlabel("Scan number")
        plt.ylabel("Normalized signal")

    def get_seed_voxel_correlations(self,brain_time_series,seed_time_series):
        """
        brain_time_series is from self.fit_timeseries\n
        seed_time_series is from self.fit_seed
        """
        seed_to_voxel_correlations = (
            np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0]
        )
        print("Seed-to-voxel correlation shape: ({}, {})".format(*seed_to_voxel_correlations.shape))

        print(f"Seed-to-voxel correlation: "
            f"min = {seed_to_voxel_correlations.min():.3f}; "
            f"max = {seed_to_voxel_correlations.max():.3f}")
        
        return seed_to_voxel_correlations
    
    def plot_seed_voxel_map(self,brain_masker,seed_to_voxel_correlations):
        """
        brain_masker is from self.mask_timeseries\n
        seed_to_voxel_correlations is from self.get_seed_voxel_correlations
        """

        seed_to_voxel_correlations_img = brain_masker.inverse_transform(
            seed_to_voxel_correlations.T
        )
        display = plotting.plot_stat_map(
            seed_to_voxel_correlations_img,
            threshold=0.5,
            vmax=1,
            cut_coords=self.seed_sphere_coords[0],
            title="Seed-to-voxel correlation (PCC seed)",
        )
        display.add_markers(
            marker_coords=self.seed_sphere_coords, marker_color="g", marker_size=300
        )

        display.savefig(self.output_dir / "pcc_seed_correlation.pdf")
    
    def save_fisher_nifti(self,brain_masker,seed_to_voxel_correlations,):

        # Apply fisher-z transform
        seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
        seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(
            seed_to_voxel_correlations_fisher_z.T
        )

        outfile = f"sub-{self.ID}_PCC_seed_correlation_z.nii.gz"

        seed_to_voxel_correlations_fisher_z_img.to_filename(
            self.output_dir / outfile
        )
    
    def get_seed_connectivity_nifti(self):
        """
        Main workflow that generates fisher z nifti files for a specified seed
        """
        chosen_run, run_idx,func_file,confound_file = self.choose_localizer_run_and_block()

        seed_masker = self.get_sphere_seed()
        seed_time_series = self.fit_seed(seed_masker,func_file,confound_file)
        brain_masker = self.mask_timeseries()
        brain_time_series = self.fit_timeseries(brain_masker,func_file,confound_file)

        seed_to_voxel_correlations = self.get_seed_voxel_correlations(
            brain_time_series,seed_time_series)

        print(f"Participant {self.ID}, run {chosen_run}, story {run_idx} used")
        self.save_fisher_nifti(brain_masker,seed_to_voxel_correlations)

if __name__ == "__main__":
    sub_id=str(sys.argv[1]) # ex. 01
    seed_con = SeedConnectivity(sub_id=sub_id)
    print("Running seed connectivity for: sub-",sub_id )
    seed_con.get_seed_connectivity_nifti()

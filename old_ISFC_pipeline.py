from nilearn.image import load_img, index_img
from brainiak.isc import isfc
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
import csv

class MovieISFCPipeline:
    def __init__(self):
        self.TR = 1.5

        # Mask Loading
        mask_dir = "dmn_roi_masks"

        self.roi_names = ["PCC_LH", "AG_LH", "MTL_LH", "mPFC_LH", "MFG_LH",
                          "PCC_RH", "AG_RH", "MTL_RH", "mPFC_RH", "MFG_RH",
                          "Hippocampus_LH","Caudate_LH","Accumbens_LH","Putamen_LH",
                          "Hippocampus_RH","Caudate_RH","Accumbens_RH","Putamen_RH"]

        self.roi_masks = {
            roi: image.load_img(os.path.join(mask_dir, f"{roi}_mask.nii.gz"))
            for roi in self.roi_names
        }
    def create_masker(self, ref_img_path):
        """
        ref_img_path: path to a representative functional image (e.g. one subject's
        preprocessed BOLD), defining the target resolution/grid for all masks.
        """
        roi_names = list(self.roi_masks.keys())

        ref_img = image.load_img(ref_img_path)
        combined_data = np.zeros(ref_img.shape[:3], dtype=int)  # [:3] in case ref_img is 4D

        for i, name in enumerate(roi_names, start=1):
            mask = self.roi_masks[name]
            mask_data = image.get_data(mask)

            if mask_data.shape != combined_data.shape:
                mask = image.resample_to_img(mask, ref_img, interpolation="nearest")
                mask_data = image.get_data(mask)

            combined_data[mask_data > 0] = i

        combined_labels_img = image.new_img_like(ref_img, combined_data)

        roi_masker = NiftiLabelsMasker(
            labels_img=combined_labels_img,
            labels=roi_names,
            smoothing_fwhm=6,
            detrend=True,
            standardize_confounds=True,
            standardize=True,
            low_pass=0.1,
            t_r=self.TR,
            memory="nilearn_cache",
            memory_level=1,
            verbose=1,
        )
        return roi_masker, roi_names

    def stable_hash(self,s):
        """For the ID to set the seed (going to need to remove soon!)"""
        return int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % (2**32)

    def get_block_bounds(self, events_path, exclude=("cartoon_introduction",)):
        """
        Reads appropriate event file and grabs the TRs corresponding to selected movie,
        excluding non-movie entries (e.g. the intro clip).
        """
        events = pd.read_csv(events_path, sep="\t")
        events = events[~events["movie_title"].isin(exclude)].reset_index(drop=True)

        blocks = []
        for _, row in events.iterrows():
            start_tr = int(row["onset"] / self.TR)
            end_tr = int((row["onset"] + row["duration"]) / self.TR)
            movie_title = row['movie_title']
            blocks.append((movie_title,start_tr, end_tr))
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

        selected_cols = motion_cols + wm_cols + csf_cols + gsr_cols

        missing = [c for c in selected_cols if c not in block_confounds.columns]
        if missing:
            raise ValueError(f"Missing expected confound columns: {missing}")
        
        block_confounds = block_confounds[selected_cols].copy()

        return block_func, block_confounds

    def build_subject_movie_timeseries(self, func_paths, confounds_paths, events_paths,
                                        roi_masker, exclude=("cartoon_introduction",),
                                        exclude_movies=()):
        """
        Returns {movie_title: cleaned_ts (n_TRs, n_regions)} for one subject,
        across both runs, skipping non-movie clips and any explicitly excluded
        movies (e.g. this subject's own localizer block).
        """
        movie_ts = {}

        for func_path, confounds_path, events_path in zip(func_paths, confounds_paths, events_paths):
            block_bounds = self.get_block_bounds(events_path, exclude=exclude)

            for movie_title, start_tr, end_tr in block_bounds:
                if movie_title in exclude_movies:
                    continue

                block_func, block_confounds = self.select_block(
                    func_path, confounds_path, [(start_tr, end_tr)], 0
                )

                ts = roi_masker.fit_transform(block_func, confounds=block_confounds)
                movie_ts[movie_title] = ts  # shape (n_TRs_this_movie, n_regions)

        return movie_ts


def stack_subjects_per_movie(all_subjects_movie_ts):
    """
    all_subjects_movie_ts: {subject_id: {movie_title: ts (n_TRs, n_regions)}}

    Returns: {movie_title: array (n_TRs, n_regions, n_subjects_included)}
    """
    movie_stacks = {}

    # collect the full set of movies across all subjects
    all_movies = set()
    for ts_dict in all_subjects_movie_ts.values():
        all_movies.update(ts_dict.keys())

    for movie_title in all_movies:
        subject_arrays = []
        for subject_id, ts_dict in all_subjects_movie_ts.items():
            if movie_title not in ts_dict:
                continue  # this subject didn't watch/have data for this movie (or it was their localizer)
            subject_arrays.append(ts_dict[movie_title])

        if len(subject_arrays) < 3:
            print(f"Skipping {movie_title}: only {len(subject_arrays)} subjects available")
            continue

        # sanity check: all subjects should have the same n_TRs for a given movie
        n_trs = subject_arrays[0].shape[0]
        assert all(arr.shape[0] == n_trs for arr in subject_arrays), \
            f"TR count mismatch across subjects for {movie_title}"

        # stack into (n_TRs, n_regions, n_subjects)
        movie_stacks[movie_title] = np.stack(subject_arrays, axis=-1)

    return movie_stacks


def sliding_window_isfc(movie_data, movie_title, window_size=15, step=1):
    """
    movie_data: array (n_TRs, n_regions, n_subjects) for ONE movie,
                already restricted to subjects who did NOT use this movie as their localizer.

    Returns: array (n_windows, n_regions, n_regions) — group-average ISFC matrix per window.
    """
    n_trs, n_regions, n_subjects = movie_data.shape
    n_windows = (n_trs - window_size) // step + 1
    print(f"Running movie: {movie_title}\n Number of Subjects: {n_subjects}")
    window_isfcs = np.zeros((n_windows, n_regions, n_regions))

    for w in range(n_windows):
        start = w * step
        end = start + window_size
        window_slice = movie_data[start:end, :, :]  # (window_size, n_regions, n_subjects)

        # isfc() with pairwise=False does leave-one-out by default,
        # returning one ISFC matrix per subject; average across subjects for the group estimate.
        # vectorize_isfcs=False keeps the full redundant (n_regions, n_regions) matrices.
        subject_isfcs = isfc(window_slice, pairwise=False, vectorize_isfcs=False)
        # subject_isfcs shape: (n_subjects, n_regions, n_regions)

        window_isfcs[w] = np.nanmean(subject_isfcs, axis=0)

    return window_isfcs


if __name__ == "__main__":
    # 1. Per subject, build the movie -> timeseries dict, excluding their own localizer movie
    all_subjects_movie_ts = {}
    subject_ids = ['01','02','07','08','09','10','11','12','13','14','16','17','18','19','20']
    loc_df = pd.read_csv("localizer_assignments.csv")

    import glob

    func_paths = {}
    confounds_paths = {}
    events_paths = {}
    bold_path = '/mnt/data1/bornstein'
    for subject_dir in sorted(glob.glob(os.path.join(bold_path,'LeeChenfMRIPrep', "sub-*"))):
        subject_id = os.path.basename(subject_dir).replace("sub-", "")
        if subject_id not in subject_ids:
            continue
        func_dir = os.path.join(subject_dir, "func")

        func_paths[subject_id] = sorted(glob.glob(
            os.path.join(func_dir, f"sub-{subject_id}_task-movie_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        ))
        confounds_paths[subject_id] = sorted(glob.glob(
            os.path.join(func_dir, f"sub-{subject_id}_task-movie_run-*_desc-confounds_timeseries.tsv")
        ))
        events_paths[subject_id] = sorted(glob.glob(
            os.path.join(bold_path,'LeeChenRaw','ds004042', f"task-movie_run-0*_events.tsv")
        ))

    pipeline = MovieISFCPipeline()
    ref_img = os.path.join(bold_path,"LeeChenfMRIPrep/sub-01/func","sub-01_task-movie_run-01_space-MNI152NLin2009cAsym_boldref.nii.gz")
    roi_masker, ordered_roi_names = pipeline.create_masker(ref_img_path=ref_img) # Just picking one reference bold image
    
    for subject_id in subject_ids:
        localizer_run = loc_df[loc_df['subject_id']==int(subject_id)]['run'].iloc[0]
        localizer_block_idx = loc_df[loc_df['subject_id']==int(subject_id)]['block_idx'].iloc[0]

        if int(localizer_run) == 1:
            movie_list = ["catch_me_if_you_can","the_record","the_boyfriend","the_shoe","keith_reynolds"]
        elif int(localizer_run) == 2:
            movie_list = ["the_rock","the_prisoner","the_black_hole","post_it_love","bus_stop"]
        print("Movie list used: ", movie_list)

        localizer_movie = movie_list[localizer_block_idx]
        print("Movie chosen: ", localizer_movie)
        print("Loading subject: ", subject_id)

        all_subjects_movie_ts[subject_id] = pipeline.build_subject_movie_timeseries(
            func_paths[subject_id], confounds_paths[subject_id], events_paths[subject_id],
            roi_masker, exclude_movies=(localizer_movie,)
        )

    movie_stacks = stack_subjects_per_movie(all_subjects_movie_ts)

    movie_window_isfcs = {}
    for movie_title, data in movie_stacks.items():
        movie_window_isfcs[movie_title] = sliding_window_isfc(data,movie_title, window_size=15, step=1)


    print("Sliding Window: ",movie_window_isfcs)
    np.savez_compressed(
        os.path.join("results","movie_full_masks_15_TR_window_isfcs.npz"),
        **{movie_title: arr for movie_title, arr in movie_window_isfcs.items()},
    )
    np.savez_compressed(
        os.path.join("results","isfcs_metadata.npz"),
        ROI_names=ordered_roi_names,
        subjects_used=subject_ids
    )

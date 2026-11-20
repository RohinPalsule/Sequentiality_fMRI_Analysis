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
from nilearn.image import resample_to_img

class FormatMVPA:
    def __init__(self):
        self.TR = 1.5

        # Mask Loading
        mask_dir = "dmn_masks/dmn_roi_masks"

        roi_masks=glob.glob(mask_dir+'/*')

        mask_names = []
        for mask in roi_masks:
            mask_names.append(mask.split('17Networks_')[1].split("_mask")[0])

        self.roi_names = mask_names

        self.roi_masks = {
            roi: image.load_img(os.path.join(mask_dir, f"17Networks_{roi}_mask.nii.gz"))
            for roi in self.roi_names
        }
        print(self.roi_masks)
        self.ref_img = os.path.join(bold_path,"LeeChenfMRIPrep/sub-01/func","sub-01_task-movie_run-01_space-MNI152NLin2009cAsym_boldref.nii.gz")
        
    def create_masker(self, roi_mask):
        """
        ref_img_path: path to a representative functional image (e.g. one subject's
        preprocessed BOLD), defining the target resolution/grid for all masks.
        """

        roi_masker = NiftiMasker(
            mask_img=roi_mask,
            detrend=True,
            standardize_confounds=True,
            standardize='zscore_sample',
            low_pass=0.1,
            t_r=self.TR,
            memory="nilearn_cache",
            memory_level=1,
            # verbose=1,
        )

        return roi_masker

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

        selected_cols = motion_cols + wm_cols + csf_cols # no gsr rn

        missing = [c for c in selected_cols if c not in block_confounds.columns]
        if missing:
            raise ValueError(f"Missing expected confound columns: {missing}")
        
        block_confounds = block_confounds[selected_cols].copy()

        return block_func, block_confounds

    def build_subject_movie_timeseries(self, func_paths, confounds_paths, events_paths, roi_masker_dict,
                                        exclude=("cartoon_introduction",),
                                        exclude_movies=()):
        """
        Returns {movie_title: {parcel: (n_TRs, n_voxels)}} for one subject,
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
                ts_per_parcel = {}

                for roi_mask_label, _ in self.roi_masks.items():
                    roi_masker = roi_masker_dict[roi_mask_label]
                    ts = roi_masker.fit_transform(block_func, confounds=block_confounds)
                    ts_per_parcel[roi_mask_label]=ts
                movie_ts[movie_title] = ts_per_parcel  # dict: keys=movie, values= dict: keys=parcel, values=timeseries at shape (n_TRs, n_voxles)
        print("Movie Timeseries: ",movie_ts)
        return movie_ts

    def create_roi_masks(self):
        roi_mask_dict = {}
        for roi_mask_label, roi_mask in self.roi_masks.items():
             roi_mask_resampled = resample_to_img(
                    roi_mask,
                    self.ref_img,
                    interpolation='nearest'
                )
             roi_masker = self.create_masker(roi_mask_resampled)
             roi_mask_dict[roi_mask_label]= roi_masker
        return roi_mask_dict


def stack_subjects_per_movie_per_parcel(all_subjects_movie_ts, min_subjects=3):
    """
    all_subjects_movie_ts: {subject_id: {movie_title: {parcel_label: (n_TRs, n_voxels)}}}

    Returns: {movie_title: {parcel_label: array (n_TRs, n_voxels, n_subjects_included)}}
    """
    movie_stacks = {}

    all_movies = set()
    for subj_dict in all_subjects_movie_ts.values():
        all_movies.update(subj_dict.keys())

    for movie_title in all_movies:
        subject_ids_with_movie = [
            sid for sid, subj_dict in all_subjects_movie_ts.items()
            if movie_title in subj_dict
        ]
        if len(subject_ids_with_movie) < min_subjects:
            print(f"Skipping {movie_title}: only {len(subject_ids_with_movie)} subjects available")
            continue

        # parcel labels are fixed by the atlas, so take the set from any subject
        parcel_labels = list(all_subjects_movie_ts[subject_ids_with_movie[0]][movie_title].keys())

        parcel_stacks = {}
        for parcel_label in parcel_labels:
            subject_arrays = []
            for sid in subject_ids_with_movie:
                arr = all_subjects_movie_ts[sid][movie_title][parcel_label]
                subject_arrays.append(arr)

            n_trs = subject_arrays[0].shape[0]
            n_voxels = subject_arrays[0].shape[1]
            for arr, sid in zip(subject_arrays, subject_ids_with_movie):
                assert arr.shape[0] == n_trs, \
                    f"TR mismatch: {sid}, {movie_title}, {parcel_label}"
                assert arr.shape[1] == n_voxels, \
                    f"Voxel count mismatch: {sid}, {movie_title}, {parcel_label} — masks may differ across subjects"

            parcel_stacks[parcel_label] = np.stack(subject_arrays, axis=-1)  # (n_TRs, n_voxels, n_subjects)

        movie_stacks[movie_title] = parcel_stacks
    print(movie_stacks)
    return movie_stacks


if __name__ == "__main__":

    all_subjects_movie_ts = {}
    subject_ids = ['01','02','07','08','09','10','11','12','13','14','16','17','18','19','20']

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

    pipeline = FormatMVPA()
    ref_img = os.path.join(bold_path,"LeeChenfMRIPrep/sub-01/func","sub-01_task-movie_run-01_space-MNI152NLin2009cAsym_boldref.nii.gz")

    roi_masker_dict = pipeline.create_roi_masks()
    for subject_id in subject_ids:
        print("Loading subject: ", subject_id)
        all_subjects_movie_ts[subject_id] = pipeline.build_subject_movie_timeseries(
            func_paths[subject_id], confounds_paths[subject_id], events_paths[subject_id], roi_masker_dict=roi_masker_dict
            #, exclude_movies=(localizer_movie,)
        )
    # print(all_subjects_movie_ts.shape)

    movie_stacks = stack_subjects_per_movie_per_parcel(all_subjects_movie_ts) # {movie_name: {parcel_name: (n_TRs, n_voxels, n_subjects)}}
    for movie_title, parcel_dict in movie_stacks.items():
        np.savez_compressed(
            os.path.join("results", f"{movie_title}_parcel_voxel_stacks.npz"),
            **parcel_dict,
        )




    # for movie_title, parcel_dict in movie_stacks.items():
    #     for parcel_name,subj_data in parcel_dict.items():
    #         for subject in subj_data.shape[2]:
    #             data = subj_data[:,:,subject] # FINALLY (n_TRs, n_voxels)

# Sequentiality fMRI Analyses

Below is the code used for the imaging analyses in the paper "Scaling up naturalistic narrative research: A reproducible large language model-based tool for evaluating contextual information gain"

Authors: Rohin A. Palsule*, Aidan Goeschel*, Jungsun Yoo, Janice Chen*, Aaron M. Bornstein*, Ari Khoudary*

DOI: https://doi.org/10.64898/2026.11.14.766702

## Running fMRI Scripts for Sequentiality

All analyses used a singularity image of python located in mypython.sif

Data was separately preprocessed using fmriPrep (parameters are in the Preprocessing folder)
Original dataset: https://github.com/OpenNeuroDatasets/ds004042.git

Data was downloaded to a private folder (available upon request) and then mounted to this repository. 

---

## Where to run each analysis

GLM Univariate Analysis with Sequentiality as the Parametric Modulator (Figure 2B): GLM/
ISFC at top 5 behavioral peaks (Figure 2C): ISFC/

Many additional analyses separate from the paper are still available in this repository, including MVPA analyses (MVPA/) and functional connectivity (ISFC/(FC Scripts) and SeedVoxel_old/) 

The Sequentiality values calculated for the continuous modulations are in sequentiality_events/ . This is what was also used to align behavioral peaks in Sequentiality.

Compiling of these scripts is still ongoing before official publication, if something is missing or broken feel free to reach out! Scripts for plotting and visualizing results will be available soon!

---

## Using Sequentiality

The metric code is available in a separate repository which can be found here: https://github.com/aidanGoesch/Sequentiality

## Questions  
If you have any questions about the code or installation, please contact me at rapalsul@uci.edu  

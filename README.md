# JASA Reproducibility Materials for "Bayesian Inference on Brain-Computer Interfaces via GLASS"

## Maintainer
- **Bangyao Zhao** ([Contact](mailto:jiankang@umich.edu))
  - Department of Biostatistics, University of Michigan

## Overview
This repository contains the code and materials to reproduce the accuracies and BCI utilities presented in the paper "Bayesian Inference on Brain-Computer Interfaces via GLASS". The reproduction focuses on the validation study using the BNCI 2014-008 Dataset, which is part of Table 1 of the supplementary materials.

## Workflow

1. **Run Calculations**
   - Execute `run_glass.py` to perform all calculations. The results will be saved in the `output` folder.
   
2. **Summarize Results**
   - Open and run `summary.ipynb` to summarize the results. This notebook will display the means and standard deviations of accuracies and BCI utilities across eight ALS subjects in the BNCI 2014-008 Dataset.

## Repository Structure

### Code
All core and helper code is located in the `code` folder.

- **`glass.py`**
  - Core code for the Gaussian Latent Channel Model with Sparse Time-Varying Effects (GLASS). This script is used as a Python module and provides the `Glass` class.
  
- **`helper_functions.py`**
  - Contains additional helper functions used in the research, such as summarizing accuracies and BCI utilities. This script is also used as a Python module in subsequent research.

### Data
The `data` folder contains scripts to manage and process the dataset.

- **`bnci2014008.py`**
  - Python module to load and process the BNCI 2014-008 Dataset. This script is designed to automatically download and process the dataset, so users do not need to manually handle the data files.
  
- **`dataset_helpers.py`**
  - Contains functions required by `bnci2014008.py` for data processing.

## Python Environment

The programming language used to reproduce this research is Python. To ensure compatibility and reproducibility, we have frozen the Python environment used at the time of the research in the `requirements.txt` file. You can set up your environment with all the necessary packages by using this file:

```bash
pip install -r requirements.txt
```


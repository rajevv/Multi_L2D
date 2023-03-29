# HAM10000 Experiments

## Download Dataset

1. Run notebook `download_ham10000dataset.ipynb`.
2. Run `python3 preprocess.py` to extrat the following files, **which we will use for training**:
   - `Data/train_data.pt`
   - `Data/validation_data.pt`
   - `Data/test_data.pt`

## Folder Structure

**Folders**

- `aistats_results`: This folder contains the final results for the AISTATS paper.
- `MLP_Mixer_model`: This folder contains the MLP Mixer model used as an expert for the HAM10000 dataset.
- `models`: This folder contains the model specification for HAM10000.

**Main files**:
The code should be run following these steps:

- `main_ham10000_oneclassifier.py`: This file trains a model for one classifier. One of the baselines for the paper.
- `main_increase_experts_hard_coded.py`: This file trains a model for the both proposed surrogate losses with the expert configuration from the paper.
<!--
- `main_increase_experts_select.py`: This file trains a model for the both proposed surrogate losses with random choices of expert configurations. **Not used in the main paper**. -->
- `hemmer_baseline_trained.py`: File to train a model using the [Hemmer et al. 2022](https://arxiv.org/abs/2206.07948) baseline.
- `validate_baselines.py`: This file returns the system accuracies for all models (including our proposed surrogate losses).
- `validate_calibration.py`: This file returns the ECE for all models (including our proposed surrogate losses).

**Others**:

- `download_ham10000dataset.ipynb`: Notebook to download HAM10000 dataset.
- `ham10000dataset.py`: Functions to load the HAM10000 dataset.
- `preprocess.py`: Processes the dataset to generate the desired format of the data into the `Data` folder, inside this HAM10000 folder.
- `conv_mixer_model.py`: Code to train the MLP Mixed Model.
- `validation_expert_model.py`: Code to valude the MLP Mixer model.

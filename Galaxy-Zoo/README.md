# Galaxy-Zoo Experiments

## Download Dataset
1. Run notebook `download_galaxyzoodataset.ipynb`.
2. Run `python3 prepare_data.py` to extrat the following files, which we will use for training:
   - `galaxy_data.pkl`: **We will work with this file.**

## Folder Structure

**Folders**
- `aistats_results`: This folder contains the final results for the AISTATS paper. 
- `models`: This folder contains the model specification for CIFAR-10.
- 
**Main files**:
The code should be run following these steps:
- `main_classifier.py`: This file trains a model for one classifier. One of the baselines for the paper. 
- `main_increase_experts_hard_coded.py`: This file trains a model for the both proposed surrogate losses with the expert configuration from the paper.
- `main_increase_experts_select.py`: This file trains a model for the both proposed surrogate losses with random choices of expert configurations. **Not used in the main paper**.
- `hemmer_baseline_trained.py`: File to train a model using the [Hemmer et al. 2022](https://arxiv.org/abs/2206.07948) baseline.
- `validate_baselines.py`: This file returns the system accuracies for all models (including our proposed surrogate losses).
- `validate_calibration.py`: This file returns the ECE for all models (including our proposed surrogate losses).


**Others**:
- `download_galaxyzoodataset.ipynb`: Notebook to download galaxy zoo dataset. 
- `galaxyzoodataset.py`: Functions to load the Galaxy-Zoo dataset.
- `prepare_data.py`: Processes the dataset to generate the desired format of the data `galaxy_data.pkl`.
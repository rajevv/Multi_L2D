# Hatespeech Experiments

## Download dataset

1. Although it's already available in this filder, you can download `labeled_data.csv` from original repository:

```
wget https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv
```

2. Process the data with `process_text_data.py`:

```
python3 process_text_data.py
```

The output will be:

- `labeled_data.csv`
- `data.pkl`: **We will work with this file.**
- `data_vectorized.pkl`
- `input_full.pkl`
- `tweets.txt`

For more information please check the original repository from Nastaran Okati et al.: (https://github.com/Networks-Learning/differentiable-learning-under-triage/tree/main/Hatespeech)

## Folder Structure

**Folders**

- `aistats_results`: This folder contains the final results for the AISTATS paper.
- `models`: This folder contains the model specification for Hatespeech.

**Main files**:
The code should be run following these steps:

- `main_classifier.py`: This file trains a model for one classifier. One of the baselines for the paper.
- `main_increase_experts.py`: This file trains a model for the both proposed surrogate losses with the expert configuration from the paper.
- `hemmer_baseline.py`: File to train a model using the [Hemmer et al. 2022](https://arxiv.org/abs/2206.07948) baseline.
- `validate.py`: This file returns the system accuracies for all models (including our proposed surrogate losses).
- `validate_calibration.py`: This file returns the ECE for all models (including our proposed surrogate losses).

**Others**:

- `download_ham10000dataset.ipynb`: Notebook to download HAM10000 dataset.
- `hatespeechdataset.py`: Functions to load the Hate Speech dataset.
- `process_text_data.py`: Processes the dataset to generate the desired format of the data.
- `conv_mixer_model.py`: Code to train the MLP Mixed Model.
- `validation_expert_model.py`: Code to valude the MLP Mixer model.

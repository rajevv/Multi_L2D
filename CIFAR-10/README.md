# CIFAR-10 Experiments

## Download Dataset
1. Run python script `download_cifar10.py`.

## Folder Structure

**Folders**
- `aistats_results`: This folder contains the final results for the AISTATS paper. 
- `conformal_experiments`: This folder contains  the conformal scripts for the CIFAR-10 experiments.
- `conformal_results`: This folder contains the results for the conformal experiments.
- `models`: This folder contains the model specification for CIFAR-10.
  
**Main files**:
The code should be run following these steps:
- `main_XX.py`: This file trains a model and saves the weights in the corresponding folder.
- `analyse_XX.py`: This file saves the confidences, expert correctness using the trained model in corresponding folder.
- `test_XX.py`: This file obtains the system accuracies and ECE from the saved data. 

**Others**:
- `cifar10dataset.py`: Functions to load the CIFAR-10 dataset.
- `download_cifar10.py`: Script to download the CIFAR-10 dataset.
## Experiments
The experiments are described below:
1. **Increase Experts**: Described in Section 7.2 from the main paper (Figure 2a, b).
2. **Random Expert**: Described in Section 7.1, Simulation #2 (Figure 2 d)).
3. **Increase Confidence**: Described in Section 7.2 (Figure 2 c).
4. **Gradual Overlap**: Described in Appendix D1.
5. **Increase Oracles No Noise**: Described in Section 7.3 (Figure 3).
6. **Increase Oracles Noise**: Described in Section 7.3 (Figure 3).
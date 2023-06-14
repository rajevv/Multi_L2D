<div align="center">

# Learning to Defer to Multiple Experts

[![Paper](https://img.shields.io/badge/paper-arxiv.2210.16955-B31B1B.svg)](https://arxiv.org/abs/2210.16955)
[![Conference](https://img.shields.io/badge/AISTATS-2023-blue)](https://arxiv.org/abs/2210.16955)

</div>

Code for Learning to Defer to Multiple Experts: Consistent Surrogate Losses, Confidence Calibration, and Conformal Ensembles [AISTATS'23]

## Installation

The installation is straightforward using the following instruction, that creates a conda virtual environment named <code>multi_l2d</code> using the provided file <code>environment.yml</code>:

```
conda env create -f environment.yml
```

## Folder Structure

- `CIFAR-10`: This folder contains the experiments and results for CIFAR-10 dataset. With respect to the AISTATS paper, it also contains experiments for **confifence calibration** [7.2] and **conformal ensembles** [7.3] (Figures 2 and 3).
- `Galaxy-Zoo`: This folder contains the experiments for Galaxy-Zoo dataset. With respect to the AISTATS paper, it also contains experiments for **overall system accuracy** [7.1] (Figure 1).
- `ham10000`: This folder contains the experiments for HAM10000 dataset. With respect to the AISTATS paper, it also contains experiments for **overall system accuracy** [7.1] (Figure 1).
- `Hatespeech`: This folder contains the experiments for Hatespeech dataset. With respect to the AISTATS paper, it also contains experiments for **overall system accuracy** [7.1] (Figure 1).
- `HMCat_ICML22`: This folder contains the results presented in the [ICML 2022 Workshop on Human-Machine Collaboration and Teaming](https://sites.google.com/view/icml-2022-hmcat/home).
- `lib`: This folder contains the shared code among al the experiments, such as the surrogate losses, the conformal methods and other utils.

## Citation

Please, if you use this code, include the following citation:

```
@inproceedings{multil2d,
  title = {Learning to Defer to Multiple Experts: Consistent Surrogate Losses, Confidence Calibration, and Conformal Ensembles},
  author = {Verma, Rajeev and Barrejon, Daniel and Nalisnick, Eric},
  booktitle = {International Conference on Artificial Intelligence and Statistics (AISTATS)},
  pages = {11415--11434},
  year = {2023},
  series = {Proceedings of Machine Learning Research},
  month = {25--27 Apr},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v206/verma23a.html},
}
```

<!---
```
@inproceedings{verma2022learning,
  title={Learning to Defer to Multiple Experts: Consistent Surrogate Losses, Confidence Calibration, and Conformal Ensembles},
  author={Verma, Rajeev and Barrej{\'o}n, Daniel and Nalisnick, Eric},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={},
  year={2023},
  organization={PMLR}
}
```
-->

## Contact

Please, do not hesitate to contact: <a href="mailto:dbarrejo@ing.uc3m.es">dbarrejo@ing.uc3m.es</a> and <a href="mailto:rajeev.ee15@gmail.com">rajeev.ee15@gmail.com</a>

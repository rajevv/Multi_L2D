import numpy as np
import torch

from baselines.data_utils import cifar
from baselines.metrics import accuracy_score

# ============================= #
# === Best Expert Baselines === #
# ============================= #
"""Functions for Evaluation of Human Baselines"""


def get_accuracy_of_best_expert(seed, expert_fns):
    NUM_EXPERTS = len(expert_fns)
    # TODO: Change to CIFAR-10
    # cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
    #                                        seed=seed, small_version=False)
    # _, _, test_loader = cifar_dl.get_data_loader()

    # Test loader
    kwargs = {'num_workers': 0, 'pin_memory': True}
    _, test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)
    test_loader = torch.utils.data.DataLoader(test_d, batch_size=1024, shuffle=False, drop_last=True, **kwargs)

    targets = torch.tensor([]).long()
    # subclass_idxs = []

    with torch.no_grad():
        # for i, (_, batch_targets, batch_subclass_idxs) in enumerate(test_loader):
        for i, (_, batch_targets) in enumerate(test_loader):
            targets = torch.cat((targets, batch_targets))
            # subclass_idxs.extend(batch_subclass_idxs)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(targets, targets))

    expert_accuracies = []
    targets = targets[:, 0]  # delete second column
    for idx in range(NUM_EXPERTS):
        preds = expert_preds[idx]
        acc = accuracy_score(targets, preds)
        expert_accuracies.append(acc)

    print(f'Best Expert Accuracy: {max(expert_accuracies)}\n')

    return max(expert_accuracies)

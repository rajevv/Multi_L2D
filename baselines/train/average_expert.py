import torch

from baselines.data_utils import cifar
from baselines.metrics import accuracy_score


class Cifar10AverageExpert:
    """A class used to represent a cohort of Cifar100Experts.

        Parameters
        ----------
        expert_fns : list[Cifar100Expert.predict]
            list of Cifar100Expert.predict functions that return the predictions of a Cifar100Expert for given subclass_idxs

        Attributes
        ----------
        expert_fns : list[Cifar100Expert.predict]
            list of Cifar100Expert.predict functions that return the predictions of a Cifar100Expert for given subclass_idxs
        num_experts : int
            the number of experts in the cohort. Is the length of expert_fns

        Methods
        -------
        predict(subclass_idxs)
            makes a prediction for the given subclass indices
        """

    def __init__(self, expert_fns=[]):
        self.expert_fns = expert_fns
        self.num_experts = len(self.expert_fns)

    def predict(self, labels):
        """Returns the predictions of a random Cifar100Expert for each image for the given subclass indices

        The first expert in expert_fns predicts the first image in subclass_idx.
        The second expert in expert_fns predicts the second image in subclass_idx.
        ...
        If all experts in expert_fns made their prediction for one image, the first expert starts again.
        If three experts are defined in expert_fns, the first expert predicts the 1st, 4th, 7th, 10th ... image

        Parameters
        ----------
        subclass_idxs : list of int
            list of subclass indices to get a prediction for

        Returns
        -------
        list of int
            returns a list of superclass indices that represent the experts prediction
        """
        all_experts_predictions = [expert_fn(input, labels) for expert_fn in self.expert_fns]
        predictions = [None] * len(labels)

        for idx, expert_predictions in enumerate(all_experts_predictions):
            predictions[idx::self.num_experts] = expert_predictions[idx::self.num_experts]

        return predictions


# ================================ #
# === Average Expert Baselines === #
# ================================ #

def get_accuracy_of_average_expert(seed, expert_fns):
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

    avg_expert = Cifar10AverageExpert(expert_fns)
    avg_expert_preds = avg_expert.predict(targets)
    targets = targets[:, 0]  # delete second column
    avg_expert_acc = accuracy_score(targets, avg_expert_preds)
    print(f'Average Expert Accuracy: {avg_expert_acc}\n')

    return avg_expert_acc

"""
From Hemmer et al. code.
Networks.
"""
import torch.nn as nn

CIFAR10_NUM_HIDDEN_UNITS = 100
CIFAR10_DROPOUT = 0.00


class CIFAR10Network(nn.Module):
    def __init__(self, output_size, softmax_sigmoid="softmax"):
        super().__init__()
        self.softmax_sigmoid = softmax_sigmoid

        self.classifier = nn.Sequential(
            nn.Dropout(CIFAR10_DROPOUT),
            # nn.Linear(512, NUM_HIDDEN_UNITS),
            nn.Linear(256, CIFAR10_NUM_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(CIFAR10_NUM_HIDDEN_UNITS, output_size)
        )

    def forward(self, features):
        # output = self.classifier(features)  # TODO: don't use for fair comparison.
        if self.softmax_sigmoid == "softmax":
            output = nn.Softmax(dim=1)(features)
        elif self.softmax_sigmoid == "sigmoid":
            output = nn.Sigmoid()(features)
        return output

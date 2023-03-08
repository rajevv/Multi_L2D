import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_clf(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):

        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedded):

        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(2) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        out = self.fc(cat)
        return out

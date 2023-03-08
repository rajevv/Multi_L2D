import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class baseline_allocator(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
				 dropout):
		
		super().__init__()
		self.convs_rej = nn.ModuleList([
									nn.Conv1d(in_channels = 1, 
											  out_channels = n_filters, 
											  kernel_size = fs) 
									for fs in filter_sizes
									])
		
		self.fc_rej = nn.Linear(len(filter_sizes) * n_filters, output_dim)
		self.dropout_rej = nn.Dropout(dropout)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, embedded):
		
		embedded = embedded.unsqueeze(1)		
		conved_rej = [F.relu(conv(embedded)).squeeze(2) for conv in self.convs_rej]					
		pooled_rej = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_rej]	
		cat_rej = self.dropout_rej(torch.cat(pooled_rej, dim = 1))
		out_rej = self.fc_rej(cat_rej)
		return self.softmax(out_rej)


class baseline_classifier(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
				 dropout):
		
		super().__init__()
						
		self.convs = nn.ModuleList([
									nn.Conv1d(in_channels = 1, 
											  out_channels = n_filters, 
											  kernel_size = fs) 
									for fs in filter_sizes
									])
		
		self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
		self.dropout = nn.Dropout(dropout)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, embedded):
		embedded = embedded.unsqueeze(1)
		conved = [F.relu(conv(embedded)).squeeze(2) for conv in self.convs]		
		pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
		cat = self.dropout(torch.cat(pooled, dim = 1))
		out = self.fc(cat)
		return self.softmax(out)


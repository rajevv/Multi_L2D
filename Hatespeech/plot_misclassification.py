import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import os
import pickle5 as pickle
from scipy import stats
import seaborn as sns
from matplotlib import rc
plt.rcParams['xtick.labelsize'] = 42
plt.rcParams['ytick.labelsize'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
plot_args = {"linestyle": "-",
                "marker": "o",
                "markeredgecolor": "k",
                "markersize": 10,
                "linewidth": 4
                }

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 30})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(file_name):
    assert(os.path.exists(file_name+'.pkl'))
    with open(file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

constraints = [0.2,0.4,0.6,0.8]
data_path = './hatespeech_data'
model_dir = './models/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12
# plt.rc('font', weight='bold')
# plt.style.use('seaborn')

def plot_misclassification_loss(data_path,machine_types):
    fig, ax = plt.subplots(figsize=(4.5,4))
    data = load_data(data_path)
    colors = {'surrogate': 'blue', 'surrogate_ova' : 'green',  'confidence': 'purple', 'score' : 'firebrick', 'differentiable': 'orange'}
    for machine_type in machine_types:
        agg_loss = data[machine_type]['agg_loss']
        print(agg_loss.shape)
        if machine_type == "Differentiable":
          machine_type = "differentiable"
        mean = np.mean(agg_loss, axis=0)
        standard_error = stats.sem(agg_loss, axis=0)
        ax.plot(constraints,mean,label=machine_type, color=colors[machine_type], **plot_args)
        ax.fill_between(constraints, mean-standard_error, mean+standard_error, alpha=0.3, color=colors[machine_type])
    ax.set_xticks(constraints)
    ax.set_ylabel(r'Classification error')
    ax.set_xlabel(r'Budget')
    ax.tick_params(axis='both', which='major')
    #ax.legend()
    ax.grid()
    plt.tight_layout()
    return fig

machine_types =  ['surrogate', 'surrogate_ova', 'score','confidence','Differentiable']
fig = plot_misclassification_loss(data_path,machine_types)
plt.show()
import os
os.makedirs('./Figs/', exist_ok=True)
name='hatespeech'
fig.savefig('./Figs/'+name+'.pdf')
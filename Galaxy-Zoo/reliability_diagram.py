from __future__ import division
#import matplotlib.pyplot as plt
#from reliability_diagrams import compute_calibration_
import torch
import numpy as np
import json
#from reliability_diagrams import _reliability_diagram_combined
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def Binarize(probs, K=10):
    temp = torch.zeros(probs.shape[0], 2)
    temp[:,0] = torch.sum(probs[:,:K], dim=1).squeeze()
    temp[:, 1] = probs[:,K].squeeze()
    return temp


class LogisticCalibration(nn.Module):
    def __init__(self):
        super(LogisticCalibration, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1)*1)
        self.m = nn.Parameter(torch.ones(1)*1)
    
    def forward(self, prob):
        temp = self.gamma*(prob - self.m)
        return torch.sigmoid(temp)

def reliability_plot(confidences, accuracies, title='test'):
    bin_data = compute_calibration(confidences, accuracies)
    fig =  _reliability_diagram_combined(bin_data, True, 'alpha',
                                         True, title, figsize=(7.5,10), 
                                         dpi=72, return_fig=True)
    plt.show()
    return fig

def _reliability_diagram_subplot(ax, bin_data, 
                                 draw_ece=True, 
                                 draw_bin_importance=False,
                                 title="Reliability Diagram", 
                                 xlabel="Confidence", 
                                 ylabel="Expected Accuracy"):
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*bin_size + 0.9*bin_size*normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    
    if draw_ece:
        ece = (bin_data["expected_calibration_error"] * 100)
        ax.text(0.98, 0.02, "ECE=%.2f" % ece, color="black", 
                ha="right", va="bottom", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # ax.legend(handles=[gap_plt, acc_plt])

def _reliability_diagram_combined(bin_data, 
                                  draw_ece, draw_bin_importance, draw_averages, 
                                  title, figsize, dpi, return_fig):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax[0], bin_data, draw_ece, draw_bin_importance, 
                                 title=title, xlabel="")

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax[1], bin_data, title="", color='blue', alpha=0.8)
    bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(np.int)
    ax[1].set_yticklabels(new_ticks)    

    # plt.show()

    if return_fig: return fig


def _confidence_histogram_subplot(ax, bin_data, color='blue', alpha=0.8,
                                  draw_averages=True,
                                  title="Examples per bin", 
                                  xlabel="Confidence",
                                  ylabel="Count"):
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    ax.bar(positions, counts, width=bin_size * 0.9, color=color, alpha=alpha)
   
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3, 
                             c="black", label="Accuracy")
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3, 
                              c="#444", label="Avg. confidence")
        ax.legend(handles=[acc_plt, conf_plt])


def reliability_diagrams(results, num_bins=10,
                         draw_ece=True, draw_bin_importance='alpha', 
                         num_cols=4, dpi=72, return_fig=True):
    """Draws reliability diagrams for one or more models."""
    ncols = 2*num_cols
    nrows = (len(results) + ncols - 1) // ncols
    nrows = 2*nrows
    figsize = (ncols * 5, nrows * 5)
    #print("print ", ncols, nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                            figsize=figsize, dpi=dpi, constrained_layout=True)

    for i, (plot_name, data) in enumerate(results.items()):
        y_conf = data["confidences"]
        y_acc = data["accuracies"]
        
        bin_data = compute_calibration(y_conf, y_acc, n_bins=num_bins)
        
        row = i // ncols
        col = i % ncols
        _reliability_diagram_subplot(ax[2*row, col], bin_data, draw_ece, 
                                        draw_bin_importance, 
                                        title="\n".join(plot_name.split()),
                                        xlabel="Confidence" if row == nrows - 1 else "",
                                        ylabel="Expected Accuracy" if col == 0 else "")
        if 'after_calibration' in plot_name:
            alpha = 0.30
            color = 'orange'
        else:
            color = 'red'
            alpha = 0.50
        _confidence_histogram_subplot(ax[2*row+1, col], bin_data, color, alpha)

    #print(i+1)
    for i in range(int(i/2), int(nrows/2) * int(ncols/2)):
        # row = i #// int(ncols/2)
        # col = i // int(ncols/2) 
        # print(row, col) 
              
        ax[i, 2].axis("off")
        ax[i, 3].axis("off")
        
        
    plt.show()

    if return_fig: return fig


def prepare(probs, labels, n_classes):
    _, pred = torch.max(probs[:,:n_classes], dim=1, keepdim=True) #want to check whether classifier is correct or not
    accuracies = pred.eq(labels) #whether the prediction made by classifier is correct or not
    confidences = torch.sum(probs[:,:n_classes], dim=1, keepdim=True) #confidence of the classifier in its prediction being correct
    return {'accuracies': accuracies, 'confidences': confidences}


def Calibration(probs, labels, n_classes, alpha):
    #binary classification problem between 1-K vs K+1
    _, pred = torch.max(probs[:,:n_classes], dim=1, keepdim=True) #want to check whether classifier is correct or not
    accuracies = pred.eq(labels) #whether the prediction made by classifier is correct or not
    confidences = torch.sum(probs[:,:n_classes], dim=1, keepdim=True) #confidence of the classifier in its prediction being correct

    #confidences = Binarize(probs)
    #print(accuracies.shape, confidences.shape)

    #plot the reliability diagram
    #fig = reliability_plot(confidences, accuracies, title = "before calibration")
    #fig.savefig('reliability_before_calibration.png')

    Criterion = nn.BCELoss()
    #print(Criterion(torch.sigmoid(confidences), accuracies.float()))
    calibrator = LogisticCalibration()
    optimizer = optim.SGD(calibrator.parameters(), lr=0.5)
    epoch_loss = []
    for iter in range(300):
        calibrator.zero_grad()
        temp = calibrator(confidences)
        #temp = temp[:,0].unsqueeze(1)
        loss = Criterion(temp, accuracies.float())
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()
    

    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.show()
    # for name in calibrator.named_parameters():
    #     print(name)
    #plot the reliability diagram
    #fig = reliability_plot(calibrator(confidences), accuracies, title="after calibration")
    #fig.savefig('reliability_after_calibration.png')
    import os
    os.makedirs('./Calibrator', exist_ok=True)
    torch.save(calibrator.state_dict(), './Calibrator/calibrator_alpha_'+str(alpha)+'.pt')
    return {'confidences': calibrator(confidences)[:,0], 'accuracies': accuracies}

        

    



def compute_calibration(confidences, accuracies, n_bins=20):
    num_bins = n_bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    #binary classification problem between 1-K vs K+1
    #_, pred = torch.max(probs[:,:n_classes], dim=1, keepdim=True) #want to check whether classifier is correct or not
    #accuracies = pred.eq(labels) #whether the prediction made by classifier is correct or not
    #confidences = torch.sum(probs[:,:n_classes], dim=1, keepdim=True) #confidence of the classifier in its prediction being correct
    ece = torch.zeros(1)

    #print(accuracies.shape, torch.mean(accuracies.float()))

    bin_accuracies = torch.zeros(num_bins)
    bin_confidences = torch.zeros(num_bins)
    bin_counts = torch.zeros(num_bins)

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        bin_counts[i] = torch.sum(in_bin.float()).item()
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            bin_accuracies[i] = accuracy_in_bin.item()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_confidences[i] = avg_confidence_in_bin.item()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    log = {'accuracies' : bin_accuracies.numpy(),
            'confidences' : bin_confidences.numpy(),
            'counts' : bin_counts.numpy(),
            'bins' : bin_boundaries,
            'expected_calibration_error' : ece, 
            'avg_accuracy' : (torch.sum(bin_accuracies*bin_counts)/torch.sum(bin_counts)).numpy(),
            'avg_confidence' : (torch.sum(bin_confidences*bin_counts)/torch.sum(bin_counts)).numpy()
            }
    return log





if __name__ == "__main__":

    log = {}
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        #print(alpha)
        with open('probs_alpha_' + str(alpha) + '.txt', 'r') as f:
            probs = json.loads(json.load(f))
        with open('true_label_alpha_' + str(alpha) + '.txt', 'r') as f:
            true_label = json.loads(json.load(f))
        
        probs_cifar = probs['2']['cifar']
        true_cifar = true_label['2']['cifar']

        if not torch.is_tensor(probs_cifar):
            probs_cifar = torch.tensor(probs_cifar)
        if not torch.is_tensor(true_cifar):
            true_cifar = torch.tensor(true_cifar).view(-1,1)
        
        log[str(alpha)] = prepare(probs_cifar, true_cifar, 10)
        log[str(alpha) + '_after_calibration'] = Calibration(probs_cifar, true_cifar, 10, alpha)
    #print(len(log))
    fig = reliability_diagrams(log, num_bins=20, num_cols=2)
    plt.show()
    fig.savefig('./calibration_for_alpha.png')
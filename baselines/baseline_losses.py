
import torch
import numpy as np
import torch.nn as nn


# CIFAR100
NUM_CLASSES = 20
DROPOUT = 0.00
NUM_HIDDEN_UNITS = 100
LR = 5e-3
USE_LR_SCHEDULER = False
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
EPOCHS = 100

# HATESPEECH
NUM_CLASSES = 2
DROPOUT = 0.00
NUM_HIDDEN_UNITS = 50
PATH_OLHS_DATA = 'labelled_data.npy'
PATH_GLOVE_MODEL = 'glove.6B.100d.txt'
NUM_EXPERTS = 20
USE_LR_SCHEDULER = True
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 5e-3
EPOCHS = 20


def joint_sparse_framework_loss(epoch, classifier_output, allocation_system_output, expert_preds, targets):
    # Input:
    #   epoch: int = current epoch (used for epoch-dependent weighting of allocation system loss)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: sigmoid outputs as expert weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size

    # loss for allocation system

    # set up zero-initialized tensor to store weighted team predictions
    batch_size = len(targets)
    weighted_team_preds = torch.zeros((batch_size, NUM_CLASSES)).to(classifier_output.device)

    # for each team member add the weighted prediction to the team prediction
    # start with machine
    weighted_team_preds = weighted_team_preds + allocation_system_output[:, 0].reshape(-1, 1) * classifier_output
    # continue with human experts
    for idx in range(NUM_EXPERTS):
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[expert_preds[idx].astype(int)]).to(classifier_output.device)
        weighted_team_preds = weighted_team_preds + allocation_system_output[:, idx + 1].reshape(-1, 1) * one_hot_expert_preds

    # calculate team probabilities using softmax
    team_probs = nn.Softmax(dim=1)(weighted_team_preds)

    # alpha2 is 1-epoch^0.5 (0.5 taken from code of preprint paper) <--- used for experiments
    alpha2 = 1 - (epoch ** -0.5)
    alpha2 = torch.tensor(alpha2).to(classifier_output.device)

    # weight the negative log likelihood loss with alpha2 to get team loss
    log_team_probs = torch.log(team_probs + 1e-7)
    allocation_system_loss = nn.NLLLoss(reduction="none")(log_team_probs, targets.long())
    allocation_system_loss = torch.mean(alpha2 * allocation_system_loss)

    # loss for classifier

    alpha1 = 1
    log_classifier_output = torch.log(classifier_output + 1e-7)
    classifier_loss = nn.NLLLoss(reduction="none")(log_classifier_output, targets.long())
    classifier_loss = alpha1 * torch.mean(classifier_loss)

    # combine both losses
    system_loss = classifier_loss + allocation_system_loss

    return system_loss

def our_loss(epoch, classifier_output, allocation_system_output, expert_preds, targets):
    # Input:
    #   epoch: int = current epoch (not used)
    #   classifier_output: softmax probabilities as class probabilities,  nxm matrix with n=batch size, m=number of classes
    #   allocation_system_output: softmax outputs as weights,  nx(m+1) matrix with n=batch size, m=number of experts + 1 for machine
    #   expert_preds: nxm matrix with expert predictions with n=number of experts, m=number of classes
    #   targets: targets as 1-dim vector with n length with n=batch_size

    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(classifier_output.device) # set up zero-initialized tensor to store team predictions
    team_probs = team_probs + allocation_system_output[:, 0].reshape(-1, 1) * classifier_output # add the weighted classifier prediction to the team prediction
    for idx in range(NUM_EXPERTS): # continue with human experts
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[expert_preds[idx].astype(int)]).to(classifier_output.device)
        team_probs = team_probs + allocation_system_output[:, idx + 1].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    system_loss = nn.NLLLoss()(log_output, targets)

    return system_loss

def mixture_of_ai_experts_loss(allocation_system_output, classifiers_outputs, targets):
    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(allocation_system_output.device)
    classifiers_outputs = classifiers_outputs.to(allocation_system_output.device)

    for idx in range(NUM_EXPERTS+1):
        team_probs = team_probs + allocation_system_output[:, idx].reshape(-1, 1) * classifiers_outputs[idx]

    log_output = torch.log(team_probs + 1e-7)
    moae_loss = nn.NLLLoss()(log_output, targets)

    return moae_loss

def mixture_of_human_experts_loss(allocation_system_output, human_expert_preds, targets):
    batch_size = len(targets)
    team_probs = torch.zeros((batch_size, NUM_CLASSES)).to(allocation_system_output.device)

    # human experts
    for idx in range(NUM_EXPERTS):
        one_hot_expert_preds = torch.tensor(np.eye(NUM_CLASSES)[human_expert_preds[idx].astype(int)]).to(allocation_system_output.device)
        team_probs = team_probs + allocation_system_output[:, idx].reshape(-1, 1) * one_hot_expert_preds

    log_output = torch.log(team_probs + 1e-7)
    mohe_loss = nn.NLLLoss()(log_output, targets)

    return mohe_loss

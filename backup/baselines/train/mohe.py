import numpy as np
import torch

from baselines.losses import mixture_of_human_experts_loss
from baselines.metrics import get_accuracy
from baselines.data_utils import cifar

# ============================ #
# === Expert Team Baseline === #
# ============================ #
"""Functions for Training and Evaluation of Mixture of Human Experts Baseline"""


def train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns, config):
    NUM_EXPERTS = len(expert_fns)
    device = config["device"]

    # switch to train mode
    feature_extractor.eval()
    allocation_system.train()

    for i, (batch_input, batch_targets, batch_subclass_idxs) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((NUM_EXPERTS, len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_subclass_idxs))

        batch_features = feature_extractor(batch_input, last_layer=True)
        batch_outputs_allocation_system = allocation_system(batch_features)

        # compute and record loss
        batch_targets = batch_targets[:, 0]
        batch_loss = mixture_of_human_experts_loss(allocation_system_output=batch_outputs_allocation_system,
                                                   human_expert_preds=expert_batch_preds, targets=batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # if USE_LR_SCHEDULER:
        #     scheduler.step()


def evaluate_mohe_one_epoch(feature_extractor, allocation_system, data_loader, expert_fns, config):
    NUM_EXPERTS = len(expert_fns)
    device = config["device"]

    feature_extractor.eval()
    allocation_system.eval()

    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    subclass_idxs = []

    with torch.no_grad():
        for i, (batch_input, batch_targets, batch_subclass_idxs) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input, last_layer=True)
            batch_allocation_system_outputs = allocation_system(batch_features)

            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets))
            # subclass_idxs.extend(batch_subclass_idxs)

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    targets = targets[:, 0]
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(targets, targets))

    # compute and record loss
    mohe_loss = mixture_of_human_experts_loss(allocation_system_output=allocation_system_outputs,
                                              human_expert_preds=expert_preds, targets=targets.long())

    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    expert_preds = expert_preds.T
    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    team_preds = expert_preds[range(len(expert_preds)), allocation_system_decisions.astype(int)]
    mohe_accuracy = get_accuracy(team_preds, targets)

    return mohe_accuracy, mohe_loss


def run_mohe(seed, expert_fns, config):
    NUM_EXPERTS = config["num_experts"]
    NUM_ClASSES = config["num_classes"]
    device = config["device"]
    EPOCHS = config["epochs"]
    LR = config["lr"]

    print(f'Training Mixture of human experts baseline')

    # === Models === TODO: Vary for each dataset

    feature_extractor = Resnet().to(device)
    allocation_system = Network(output_size=NUM_EXPERTS,
                                softmax_sigmoid="softmax").to(device)
    # TODO: Change to CIFAR-10
    # cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
    #                                        seed=seed, small_version=False)
    # train_loader, val_loader, test_loader = cifar_dl.get_data_loader()


    # === Data === TODO: Vary for each dataset
    trainD, valD = cifar.read(test=False, only_id=True, data_aug=True)
    _, test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)
    # Train / Val loaders
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(trainD,
                                               batch_size=1024, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(valD,
                                             batch_size=1024, shuffle=True, drop_last=True, **kwargs)
    # Test loader
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_d, batch_size=1024, shuffle=False, drop_last=True, **kwargs)

    parameters = allocation_system.parameters()
    # optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.SGD(parameters, LR,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

    best_val_system_loss = 100
    best_test_system_accuracy = None

    for epoch in range(1, EPOCHS + 1):
        print("-" * 20, f'Epoch {epoch}', "-" * 20)

        train_mohe_one_epoch(feature_extractor, allocation_system, train_loader, optimizer, scheduler, expert_fns)
        val_mohe_accuracy, val_mohe_loss = evaluate_mohe_one_epoch(feature_extractor, allocation_system, val_loader,
                                                                   expert_fns)
        test_mohe_accuracy, test_mohe_loss = evaluate_mohe_one_epoch(feature_extractor, allocation_system, test_loader,
                                                                     expert_fns)

        if val_mohe_loss < best_val_system_loss:
            best_val_system_loss = val_mohe_loss
            best_test_system_accuracy = test_mohe_accuracy

    print(f'Mixture of Human Experts Accuracy: {best_test_system_accuracy}\n')
    return best_test_system_accuracy

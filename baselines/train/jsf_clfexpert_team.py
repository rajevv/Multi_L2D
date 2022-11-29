import numpy as np
import torch
from tqdm import tqdm

from baselines.losses import joint_sparse_framework_loss, our_loss
from baselines.data_utils import cifar
from baselines.models.wideresnet import WideResNet
from baselines.metrics import get_metrics
# ==================================== #
# === Keswani and Hemmer Baselines === #
# ==================================== #
"""Functions for Training and Evaluation of Our Approach and JSF"""


def train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler,
                    expert_fns, loss_fn, config):
    NUM_EXPERTS = config["num_experts"]
    device = config["device"]

    feature_extractor.eval()
    classifier.train()
    allocation_system.train()

    # for i, (batch_input, batch_targets, batch_subclass_idxs) in enumerate(train_loader):
    for i, (batch_input, batch_targets) in enumerate(train_loader):

        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        expert_batch_preds = np.empty((NUM_EXPERTS, len(batch_targets)))
        for idx, expert_fn in enumerate(expert_fns):
            expert_batch_preds[idx] = np.array(expert_fn(batch_input, batch_targets))

        batch_targets = batch_targets[:, 0]  # Delete column 2
        batch_features = feature_extractor(batch_input, last_layer=True)
        batch_outputs_classifier = classifier(batch_features)
        batch_outputs_allocation_system = allocation_system(batch_features)

        batch_loss = loss_fn(epoch=epoch, classifier_output=batch_outputs_classifier,
                             allocation_system_output=batch_outputs_allocation_system,
                             expert_preds=expert_batch_preds, targets=batch_targets)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # if USE_LR_SCHEDULER:
        #     scheduler.step()


def evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, data_loader, expert_fns, loss_fn, config):
    NUM_EXPERTS = config["num_experts"]
    device = config["device"]

    feature_extractor.eval()
    classifier.eval()
    allocation_system.eval()

    classifier_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    subclass_idxs = []

    with torch.no_grad():
        for i, (batch_input, batch_targets) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input)
            batch_classifier_outputs = classifier(batch_features)
            batch_allocation_system_outputs = allocation_system(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets))

    expert_preds = np.empty((NUM_EXPERTS, len(targets)))
    for idx, expert_fn in enumerate(expert_fns):
        expert_preds[idx] = np.array(expert_fn(targets, targets))

    classifier_outputs = classifier_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets[:, 0]  # Delete column 2
    targets = targets.cpu().numpy()

    system_accuracy, system_loss, metrics = get_metrics(epoch, allocation_system_outputs, classifier_outputs,
                                                        expert_preds, targets, loss_fn)

    return system_accuracy, system_loss, metrics


def run_team_performance_optimization(method, seed, expert_fns, config):
    NUM_EXPERTS = config["num_experts"]
    NUM_CLASSES = config["num_classes"]
    device = config["device"]
    EPOCHS = config["epochs"]
    LR = config["lr"]

    print(f'Team Performance Optimization with {method}')

    if method == "Joint Sparse Framework":
        loss_fn = joint_sparse_framework_loss
        allocation_system_activation_function = "sigmoid"


    elif method == "Our Approach":
        loss_fn = our_loss
        allocation_system_activation_function = "softmax"

    # === Models === TODO: Vary for each dataset
    # feature_extractor = Resnet().to(device)
    feature_extractor = WideResNet(28, 3, NUM_CLASSES, 4, dropRate=0.0).to(device)
    classifier = Network(output_size=NUM_CLASSES,
                         softmax_sigmoid="softmax").to(device)
    allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                softmax_sigmoid=allocation_system_activation_function).to(device)

    # === Data === TODO: Vary for each dataset
    # cifar_dl = CIFAR100_3_Split_Dataloader(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
    #                                        seed=seed, small_version=False)
    # train_loader, val_loader, test_loader = cifar_dl.get_data_loader()

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

    parameters = list(classifier.parameters()) + list(allocation_system.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.SGD(parameters, LR,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

    best_val_system_accuracy = 0
    best_val_system_loss = 100
    best_metrics = None

    for epoch in tqdm(range(1, EPOCHS + 1)):
        train_one_epoch(epoch, feature_extractor, classifier, allocation_system, train_loader, optimizer, scheduler,
                        expert_fns, loss_fn)

        val_system_accuracy, val_system_loss, _ = evaluate_one_epoch(epoch, feature_extractor, classifier,
                                                                     allocation_system, val_loader, expert_fns, loss_fn)
        _, _, test_metrics = evaluate_one_epoch(epoch, feature_extractor, classifier, allocation_system, test_loader,
                                                expert_fns, loss_fn)

        if method == "Joint Sparse Framework":
            if val_system_accuracy > best_val_system_accuracy:
                best_val_system_accuracy = val_system_accuracy
                best_metrics = test_metrics

        elif method == "Our Approach":
            if val_system_loss < best_val_system_loss:
                best_val_system_loss = val_system_loss
                best_metrics = test_metrics

    print(f'\n Earlystopping Results for {method}:')
    system_metrics_keys = [key for key in best_metrics.keys() if "System" in key]
    for k in system_metrics_keys:
        print(f'\t {k}: {best_metrics[k]}')
    print()

    classifier_metrics_keys = [key for key in best_metrics.keys() if "Classifier" in key]
    for k in classifier_metrics_keys:
        print(f'\t {k}: {best_metrics[k]}')
    print()

    """for exp_idx in range(NUM_EXPERTS):
      expert_metrics_keys = [key for key in best_metrics.keys() if f'Expert {exp_idx+1} ' in key]
      for k in expert_metrics_keys:
          print(f'\t {k}: {best_metrics[k]}')
    print()"""

    return best_metrics["System Accuracy"], best_metrics["Classifier Coverage"]


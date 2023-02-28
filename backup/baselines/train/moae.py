import numpy as np
import torch

from baselines.data_utils import cifar
from baselines.losses import mixture_of_ai_experts_loss
from baselines.metrics import get_accuracy

# ================================ #
# === Classifier Team Baseline === #
# ================================ #
"""Functions for Training and Evaluation of Mixture of Artificial Experts Baseline"""


def train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler, config):
    NUM_EXPERTS = config["num_experts"]
    NUM_CLASSES = config["num_classes"]
    device = config["device"]

    # switch to train mode
    feature_extractor.eval()
    allocation_system.train()
    for classifier in classifiers:
        classifier.train()

    for i, (batch_input, batch_targets, _) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        batch_features = feature_extractor(batch_input, last_layer=True)
        batch_outputs_allocation_system = allocation_system(batch_features)
        batch_outputs_classifiers = torch.empty((NUM_EXPERTS + 1, len(batch_targets), NUM_CLASSES))
        for idx, classifier in enumerate(classifiers):
            batch_outputs_classifiers[idx] = classifier(batch_features)

        # compute and record loss
        batch_targets = batch_targets[:, 0]
        batch_loss = mixture_of_ai_experts_loss(allocation_system_output=batch_outputs_allocation_system,
                                                classifiers_outputs=batch_outputs_classifiers, targets=batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # if USE_LR_SCHEDULER:
        #     scheduler.step()


def evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system, data_loader, config):
    NUM_EXPERTS = config["num_experts"]
    NUM_CLASSES = config["num_classes"]
    device = config["device"]

    feature_extractor.eval()
    allocation_system.eval()
    for classifier in classifiers:
        classifier.eval()

    classifiers_outputs = torch.tensor([]).to(device)
    allocation_system_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).long().to(device)

    with torch.no_grad():
        for i, (batch_input, batch_targets, _) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_features = feature_extractor(batch_input, last_layer=True)
            batch_allocation_system_outputs = allocation_system(batch_features)
            batch_outputs_classifiers = torch.empty((NUM_EXPERTS + 1, len(batch_targets), NUM_CLASSES)).to(device)
            for idx, classifier in enumerate(classifiers):
                batch_outputs_classifiers[idx] = classifier(batch_features)

            classifiers_outputs = torch.cat((classifiers_outputs, batch_outputs_classifiers), dim=1)
            allocation_system_outputs = torch.cat((allocation_system_outputs, batch_allocation_system_outputs))
            targets = torch.cat((targets, batch_targets.float()))

    targets = targets[:, 0]
    moae_loss = mixture_of_ai_experts_loss(allocation_system_output=allocation_system_outputs,
                                           classifiers_outputs=classifiers_outputs, targets=targets.long())

    classifiers_outputs = classifiers_outputs.cpu().numpy()
    allocation_system_outputs = allocation_system_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    allocation_system_decisions = np.argmax(allocation_system_outputs, 1)
    classifiers_preds = np.argmax(classifiers_outputs, 2).T
    team_preds = classifiers_preds[range(len(classifiers_preds)), allocation_system_decisions.astype(int)]
    moae_accuracy = get_accuracy(team_preds, targets)

    return moae_accuracy, moae_loss


def run_moae(seed, config):
    NUM_EXPERTS = config["num_experts"]
    NUM_CLASSES = config["num_classes"]
    device = config["device"]
    EPOCHS = config["epochs"]
    LR = config["lr"]

    print(f'Training Mixture of artificial experts baseline')

    # === Models === TODO: Vary for each dataset
    feature_extractor = Resnet().to(device)
    allocation_system = Network(output_size=NUM_EXPERTS + 1,
                                softmax_sigmoid="softmax").to(device)
    classifiers = []
    for _ in range(NUM_EXPERTS + 1):
        classifier = Network(output_size=NUM_CLASSES,
                             softmax_sigmoid="softmax").to(device)
        classifiers.append(classifier)

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

    parameters = list(allocation_system.parameters())
    for classifier in classifiers:
        parameters += list(classifier.parameters())

    # optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.SGD(parameters, LR,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

    best_val_system_loss = 100
    best_test_system_accuracy = None

    for epoch in range(1, EPOCHS + 1):
        print("-" * 20, f'Epoch {epoch}', "-" * 20)

        train_moae_one_epoch(feature_extractor, classifiers, allocation_system, train_loader, optimizer, scheduler)
        val_moae_accuracy, val_moae_loss = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system,
                                                                   val_loader)
        test_moae_accuracy, test_moae_loss = evaluate_moae_one_epoch(feature_extractor, classifiers, allocation_system,
                                                                     test_loader)

        if val_moae_loss < best_val_system_loss:
            best_val_system_loss = val_moae_loss
            best_test_system_accuracy = test_moae_accuracy

    print(f'Mixture of Artificial Experts Accuracy: {best_test_system_accuracy}\n')
    return best_test_system_accuracy

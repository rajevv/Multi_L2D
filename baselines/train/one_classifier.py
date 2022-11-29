import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from baselines.data_utils import cifar
from baselines.metrics import get_accuracy
from baselines.models.wideresnet import WideResNet

# =============================== #
# === One Classifier Baseline === #
# =============================== #
"""Functions for Training and Evaluation of Full Automation Baseline"""


def train_full_automation_one_epoch(model, train_loader, optimizer, scheduler, config):
    device = config["device"]

    # switch to train mode

    model.train()

    for i, (batch_input, batch_targets) in enumerate(train_loader):
        batch_input = batch_input.to(device)
        batch_targets = batch_targets.to(device)

        batch_outputs_classifier = model(batch_input)
        # batch_outputs_classifier = classifier(batch_features)

        log_output = torch.log(batch_outputs_classifier + 1e-7)
        batch_targets = batch_targets[:, 0]
        batch_loss = nn.NLLLoss()(log_output, batch_targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # if USE_LR_SCHEDULER:
        #     scheduler.step()


def evaluate_full_automation_one_epoch(model, data_loader, config):
    device = config["device"]

    model.eval()

    classifier_outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    filenames = []

    with torch.no_grad():
        for i, (batch_input, batch_targets,) in enumerate(data_loader):
            batch_input = batch_input.to(device)
            batch_targets = batch_targets.to(device)

            batch_classifier_outputs = model(batch_input)
            # batch_classifier_outputs = classifier(batch_features)

            classifier_outputs = torch.cat((classifier_outputs, batch_classifier_outputs))
            targets = torch.cat((targets, batch_targets))

    log_output = torch.log(classifier_outputs + 1e-7)
    targets = targets[:, 0]
    full_automation_loss = nn.NLLLoss()(log_output, targets.long())

    classifier_outputs = classifier_outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    classifier_preds = np.argmax(classifier_outputs, 1)
    full_automation_accuracy = get_accuracy(classifier_preds, targets)

    return full_automation_accuracy, full_automation_loss


def run_full_automation(seed, config):
    NUM_CLASSES = config["num_classes"]
    device = config["device"]
    EPOCHS = config["epochs"]
    LR = config["lr"]

    print(f'Training full automation baseline')

    # feature_extractor = Resnet().to(device)
    model = WideResNet(28, 3, NUM_CLASSES, 4, dropRate=0.0).to(device)

    # classifier = Network(output_size=NUM_CLASSES,
    #                      softmax_sigmoid="softmax").to(device)

    # TODO: Change to CIFAR10
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

    # optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), LR,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader))

    best_val_system_loss = 100
    best_test_system_accuracy = None

    for epoch in tqdm(range(1, EPOCHS + 1)):
        train_full_automation_one_epoch(model, train_loader, optimizer, scheduler)

        val_system_accuracy, val_system_loss = evaluate_full_automation_one_epoch(model,
                                                                                  val_loader)
        test_system_accuracy, test_system_loss, = evaluate_full_automation_one_epoch(model,
                                                                                     test_loader)

        if val_system_loss < best_val_system_loss:
            best_val_system_loss = val_system_loss
            best_test_system_accuracy = test_system_accuracy

        print("Val Acc:{} | Test Acc: {}".format(val_system_accuracy, test_system_accuracy))

    print(f'Full Automation Accuracy: {best_test_system_accuracy}\n')
    return best_test_system_accuracy

import torch
import torchvision
import torch.nn as nn


class Resnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.resnet = torchvision.models.resnet18(pretrained=True)  # CIFAR-10 Hemmer
        self.resnet = torchvision.models.resnet50(
            pretrained=True)  # Galaxy-Zoo Ours
        del self.resnet.fc

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.training = False

    def forward(self, x):
        # Hemmer et al ===
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)

        features = torch.flatten(x, 1)
        return features


class Network(nn.Module):
    def __init__(self, output_size, softmax_sigmoid="softmax"):
        super().__init__()
        self.softmax_sigmoid = softmax_sigmoid
        DROPOUT = 0  # as Hemmer et al
        NUM_HIDDEN_UNITS = 100  # as Hemmer et al

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(2048, NUM_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_UNITS, output_size)
        )

    def forward(self, features):
        output = self.classifier(features)
        if self.softmax_sigmoid == "softmax":
            output = nn.Softmax(dim=1)(output)
        elif self.softmax_sigmoid == "sigmoid":
            output = nn.Sigmoid()(output)
        return output

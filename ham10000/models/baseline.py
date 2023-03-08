import torch
import torch.nn as nn
import torchvision


class ResNet34(torch.nn.Module):
    def __init__(self, train_weights=False):
        super().__init__()

        self.resnet = torchvision.models.resnet34(pretrained=True)
        del self.resnet.fc

        for param in self.resnet.parameters():
            param.requires_grad = train_weights

        self.training = train_weights

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

class ResNet34_oneclf(nn.Module):
    def __init__(self, out_size):
        super(ResNet34_oneclf, self).__init__()
        self.resnet34 = torchvision.models.resnet34(pretrained=True)
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size))
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet34(x)  # out shape: Bs x 512
        return x


class Network(nn.Module):
    def __init__(self, output_size, softmax_sigmoid="softmax"):
        super().__init__()
        self.softmax_sigmoid = softmax_sigmoid
        DROPOUT = 0  # as Hemmer et al
        NUM_HIDDEN_UNITS = 100  # as Hemmer et al

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(512, output_size)
        )
        #     nn.ReLU(),
        #     nn.Linear(NUM_HIDDEN_UNITS, output_size)
        # )

    def forward(self, features):
        output = self.classifier(features)
        if self.softmax_sigmoid == "softmax":
            output = nn.Softmax(dim=1)(output)
        elif self.softmax_sigmoid == "sigmoid":
            output = nn.Sigmoid()(output)
        return output

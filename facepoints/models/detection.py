"""Convolutional regression model for 14 facial landmarks."""

import torch.nn as nn


class MyDetectionModel(nn.Module):
    """CNN with a four-stage convolutional path and a parallel skip branch."""

    def __init__(self, input_image_shape, num_layers=None):
        super().__init__()
        height, width = input_image_shape
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.norm1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        height = (height - 2) // 2
        width = (width - 2) // 2

        self.conv2 = nn.Conv2d(64, 128, 3)
        self.norm2 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)
        height = (height - 2) // 2
        width = (width - 2) // 2

        self.conv3 = nn.Conv2d(128, 256, 3)
        self.norm3 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)
        height = (height - 2) // 2
        width = (width - 2) // 2

        self.conv4 = nn.Conv2d(256, 512, 3)
        self.norm4 = nn.BatchNorm2d(num_features=512, momentum=0.9)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.4)
        height = (height - 2) // 2
        width = (width - 2) // 2

        if height < 1 or width < 1:
            raise ValueError(f"Input image shape {input_image_shape} is too small for the model.")
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(height * width * 512, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.relu_dense = nn.ReLU()
        self.dense2 = nn.Linear(128, 28)

        self.skip_block = nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.BatchNorm2d(num_features=3, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 3),
            nn.BatchNorm2d(num_features=3, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 3),
            nn.BatchNorm2d(num_features=3, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 512, 3),
            nn.BatchNorm2d(num_features=512, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, inputs):
        x_skip = self.skip_block(inputs)

        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout2(x)

        x = self.flatten(x + x_skip)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.relu_dense(x)
        return self.dense2(x)

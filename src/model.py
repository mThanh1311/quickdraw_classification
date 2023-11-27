import torch
import torch.nn as nn
from torchsummary import summary

class QD_Model(nn.Module):
    def __init__(self, input_size=28, num_classes=20):
        super(QD_Model, self).__init__()
        self.num_classes = num_classes
        self.conv1 = self.cnn_block(in_channels=1, out_channels=32)
        self.conv2 = self.cnn_block(in_channels=32, out_channels=64)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def cnn_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, image):
        output = self.conv1(image)
        output = self.conv2(output)

        # [batch_size, channels, height, width] --> [batch, channels * height * weight]
        output = output.view(output.size(0), -1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

if __name__ == '__main__':
    model = QD_Model()
    model.train()
    #summary(model, (1,28,28))
    sample_input = torch.randn(4, 1, 28, 28)
    result = model(sample_input)
    print(result.shape)
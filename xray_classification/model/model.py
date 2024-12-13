import torch.nn as nn
import torch.nn.functional as F


class XRayClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)  # N * 16 * 80 * 80
        self.pool1 = nn.MaxPool2d(2, 2)  # N * 16 * 40 * 40
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)  # N * 32 * 40 * 40
        self.pool2 = nn.MaxPool2d(2, 2)  # N * 32 * 20 * 20
        self.fc1 = nn.Linear(32 * 20 * 20, 1024)  # N * 1024
        self.fc2 = nn.Linear(1024, 1)  # N * 1

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # N * 16 * 40 * 40
        x = self.pool2(F.relu(self.conv2(x)))  # N * 32 * 20 * 20
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    import torch
    model = XRayClassifier()
    x = torch.randn(1, 3, 80, 80)
    y = model(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")

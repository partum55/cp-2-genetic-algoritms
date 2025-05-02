import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):

    def __init__(self, device):
        super(CNN, self).__init__()

        self.device = device
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # batch norm
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # maxpooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully conected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def evaluate(self, data_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def get_flat_params(self):
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.clone()

    def set_flat_params(self, flat_params):
        pointer = 0
        for param in self.parameters():
            num_params = param.numel()
            param.data.copy_(flat_params[pointer : pointer + num_params].view_as(param))
            pointer += num_params

    @classmethod
    def create_random_model(cls, device):
        model = cls(device).to(device)

        for param in model.parameters():
            param.data = torch.randn_like(param)

        return model

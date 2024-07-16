import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  def __init__(self, n_channels=3, n_classes=10):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
    self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv3_a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.conv3_b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv4_a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.conv4_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv5_a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.conv5_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc = nn.Linear(512, n_classes) # 224 * 224

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.maxpool1(x)
    x = F.relu(self.conv2(x))
    x = self.maxpool2(x)
    x = F.relu(self.conv3_a(x))
    x = F.relu(self.conv3_b(x))
    x = self.maxpool3(x)
    x = F.relu(self.conv4_a(x))
    x = F.relu(self.conv4_b(x))
    x = self.maxpool4(x)
    x = F.relu(self.conv5_a(x))
    x = F.relu(self.conv5_b(x))
    x = self.maxpool5(x)
    x = x.view(-1, 512)  
    out = self.fc(x)
    return out
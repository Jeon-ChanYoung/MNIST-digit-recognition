import torch.nn as nn

# 10회 학습세트 : 98.31% | 테스트세트 " 98.85%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential( #입력 : (1, 28, 28) 흑백, 28x28
            nn.Conv2d(1, 64, (3,3)), #(1, 28, 28) -> (64, 26, 26) parameter = 1X64 x (3x3 + 1) : 640
            nn.ReLU(),
            nn.MaxPool2d((2, 2)), #(64, 26, 26) -> (64, 13, 13)
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, (3,3)), #(64, 13, 13) -> (128, 11, 11) parameter = 64X128 X (3X3 + 1) : 73856
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   #(128, 11, 11) -> (128, 5, 5)
            nn.Dropout(0.5)
        )
        self.layer3 = nn.Linear(128*5*5, 128) #parameter = 128x5x5x128 ....
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten: (128, 5, 5) -> (128*5*5)
        x = self.layer3(x)
        x = self.output(x)
        return x

# 10회 학습세트 : 99.2300% | 테스트세트 " 98.8718% (lr = 0.0001)
# 10회 학습세트 : 99.7850% | 테스트세트 " 99.0116% (lr = 0.001)
class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # assuming input images are 28x28 pixels
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.classifier(x)
        return x

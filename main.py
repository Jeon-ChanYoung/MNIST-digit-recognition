from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
from CNN_model import CNN
import numpy as np
import torch.nn as nn
import torch.optim as optim


# MNIST 데이터셋 로드
PATH = "./my_deepLearning"
train_dataset = datasets.MNIST(PATH, train=True, download=True)
test_dataset = datasets.MNIST(PATH, train=False, download=True)

# train_dataset.data -> 28x28크기의 숫자 Tensor자료 6만장 (60000, 28, 28)
# train_dataset.targets -> 정답 데이터 tensor([5, 0, 4,  ..., 5, 6, 8])
# 255로 나누는 이유는 정규화 (각 밝기가 0~255로 나타나있기에 255로 나누어 0~1사이의 수로 표현)
X_train = train_dataset.data / 255 
X_test = test_dataset.data / 255
y_train = train_dataset.targets
y_test = test_dataset.targets

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


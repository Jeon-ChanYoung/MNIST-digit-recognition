from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
from CNN_model import CNN, CNN1
import numpy as np
import torch
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

#2차원 (높이, 너비)에서 3차원 (채널, 높이, 너비)형태로 변환
X_train, X_test = X_train.unsqueeze(1), X_test.unsqueeze(1)

train_dset = TensorDataset(X_train, y_train)
test_dset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

model = CNN1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model:CNN, optimizer, criterion, loader):
    accuracy = 0
    model.train()

    for X,y in loader:
        optimizer.zero_grad()
        hypothesis = model(X)
        loss = criterion(hypothesis, y)
        loss.backward()
        optimizer.step()
        predict = torch.argmax(hypothesis, 1)
        acc = (predict == y).float().mean()
        accuracy += acc.item()
    return accuracy / len(loader)

def evaluate(model:CNN, loader):
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for X,y in loader:
            hypothesis = model(X)
            predict = torch.argmax(hypothesis, 1)
            acc = (predict == y).float().mean()
            accuracy += acc.item()
        return accuracy / len(loader)


for epoch in range(1, 11):
    train_accuracy = train(model, optimizer, criterion, train_loader)
    test_accuracy = evaluate(model, test_loader)
    print(f'{epoch}회 학습세트 : {train_accuracy*100:.4f}% | 테스트세트 " {test_accuracy*100:.4f}%')

torch.save(model.state_dict(), './my_deepLearning/model_parameter/CNN_Model2_MNIST.pt')
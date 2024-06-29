from torchvision import datasets

# MNIST 데이터셋 로드
PATH = "./my_deepLearning"
train_dataset = datasets.MNIST(PATH, train=True, download=True)
test_dataset = datasets.MNIST(PATH, train=False, download=True)

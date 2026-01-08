from medmnist import DermaMNIST

# 下载 28x28 的版本用于快速测试
train_dataset = DermaMNIST(split="train", download=True, size=28)

train_dataset_high = DermaMNIST(split="train", download=True, size=224)

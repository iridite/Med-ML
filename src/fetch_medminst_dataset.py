import os

from medmnist import DermaMNIST


def check_data():
    print("--- 开始处理 28x28 版本 ---")
    # size=28 是默认值
    train_28 = DermaMNIST(split="train", download=True, size=28)
    print(f"28x28 数据集加载成功！样本量: {len(train_28)}")

    print("\n--- 开始处理 224x224 版本 ---")

    # 显式下载 224 版本
    try:
        train_224 = DermaMNIST(split="train", download=True, size=224)
        print(f"224x224 数据集加载成功！样本量: {len(train_224)}")

        # 打印一张图片的形状确认一下
        img, target = train_224[0]
        print(f"单张图片形状: {img.size}, 标签: {target}")

    except Exception as e:
        print(f"下载/加载失败，错误原因: {e}")


if __name__ == "__main__":
    check_data()

import os


class Config:
    # === 1. 路径设置 (请根据你实际的解压位置修改) ===
    # 假设你的项目目录是 ~/Melanoma_Causal_MVP
    # 数据在 ~/Melanoma_Causal_MVP/data/raw
    DATA_DIR = "./data/siim-isic-melanoma-classification"

    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    # 注意：Kaggle 解压后可能是 train/ 文件夹，也可能是 jpeg/train/
    # 请你去文件夹里确认一下图片真正所在的目录名字，并修改下面这行
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "jpeg/train")

    # === 2. 模型参数 ===
    IMG_SIZE = 224  # MVP 阶段使用 224x224 加速
    BATCH_SIZE = 32  # 如果显存不够，改小到 16 或 8
    EPOCHS = 10  # 先跑 5-10 轮验证代码
    LR = 3e-4  # 学习率

    # === 3. 随机种子 (复现性) ===
    SEED = 42

    # === 4. 硬件 ===
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

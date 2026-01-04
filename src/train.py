import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 导入我们自己写的模块
from src.config import Config
from src.dataset import MelanomaDataset, get_preprocessed_df
from src.models import CausalFusionModel
from src.utils import seed_everything


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    # 进度条
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Train]")

    for images, metas, targets, weights in pbar:
        # 数据移到 GPU
        images, metas = images.to(device), metas.to(device)
        targets = targets.to(device).unsqueeze(1)  # [batch] -> [batch, 1]
        weights = weights.to(device).unsqueeze(1)  # [batch] -> [batch, 1]

        # 前向传播
        optimizer.zero_grad()
        logits = model(images, metas)

        # === 计算损失 (支持因果加权) ===
        # criterion 设定为 none，返回每个样本的loss
        raw_loss = criterion(logits, targets)

        # 手动乘以权重 (如果 weights 全是 1，这里就等于没变)
        loss = (raw_loss * weights).mean()

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, metas, targets, _ in tqdm(loader, desc="[Valid]"):
            images, metas = images.to(device), metas.to(device)
            targets_gpu = targets.to(device).unsqueeze(1)

            logits = model(images, metas)

            # Loss
            loss = criterion(logits, targets_gpu)
            val_loss += loss.item()

            # 记录结果用于计算 AUC
            all_targets.extend(targets.numpy())
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())  # Logits转概率

    # 计算 AUC (Area Under Curve)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5  # 防止只有一个类别时报错

    return val_loss / len(loader), auc


def main():
    seed_everything(Config.SEED)
    print(f"使用的设备: {Config.DEVICE}")

    # 1. 数据准备
    df = get_preprocessed_df(Config.TRAIN_CSV)

    # 简单的图像增强 (Resize -> Normalize -> Tensor)
    transforms_train = A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),  # 随机水平翻转
            A.VerticalFlip(p=0.5),  # 随机垂直翻转
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    transforms_val = A.Compose(
        [A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), A.Normalize(), ToTensorV2()]
    )

    full_dataset = MelanomaDataset(df, Config.TRAIN_IMG_DIR, transform=transforms_train)

    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 这里的 val_ds transform 其实还没覆盖，为了MVP代码简洁先共用
    # (更严谨的做法是重写 Dataset wrapper 覆盖 transform，先跳过)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 2. 模型初始化
    # 注意：表格维度从 full_dataset 自动获取
    meta_dim = full_dataset.meta_features.shape[1]
    model = CausalFusionModel(meta_features_dim=meta_dim).to(Config.DEVICE)

    # 3. 优化器和损失
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)

    # 重要：BCEWithLogitsLoss 自带 Sigmoid，且 reduction='none' 配合我们的加权逻辑
    # 也可以在这里加上 pos_weight=torch.tensor([10.0]) 来应对数据不平衡
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # 4. 训练循环
    best_auc = 0.0
    for epoch in range(Config.EPOCHS):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, epoch
        )

        # 验证
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # 保存最好的模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "./saved_models/best_model.pth")
            print("🚀 新的最佳模型已保存！")

    print("✅ 训练结束！")


if __name__ == "__main__":
    main()

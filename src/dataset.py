import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        参数:
            df (pd.DataFrame): 预处理好的一整张表格
            img_dir (str): 存放图片的文件夹路径
            transform (albumentations): 图像增强/变换函数
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        # 提取 image_name (不需要扩展名，CSV里通常没有 .jpg)
        self.image_ids = self.df["image_name"].values
        # 提取 target
        self.targets = self.df["target"].values

        # === 修改点在这里 ===
        # 我们必须把字符串列（诊断结果）和我们试图预测的标签排除掉
        # diagnosis: 具体诊断结果（包含文本）
        # benign_malignant: 良性还是恶性（包含文本，且涉及泄漏）
        ignore_cols = [
            "image_name",
            "patient_id",
            "target",
            "causal_weight",
            "diagnosis",
            "benign_malignant",  # <--- 新增这俩坏蛋
        ]

        # 剩下的列才会作为 features 进入 MLP
        self.meta_cols = [c for c in self.df.columns if c not in ignore_cols]
        self.meta_features = self.df[self.meta_cols].values

        # 我们可以打印一下现在真正留下来的是哪些列，一定是纯数字的
        print(f"检测到有效表格特征: {len(self.meta_cols)} 列 -> {self.meta_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 拼凑图片路径
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        # 2. 读取图片 (使用 OpenCV)
        image = cv2.imread(img_path)
        if image is None:
            # 防御性编程：防止找不到图报错停止训练，给一张黑图（实战中可以记录Log）
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 图像增强/转换
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 4. 准备表格特征 (转 Tensor)
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)

        # 5. 准备 Label
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        # === 【因果推断预留接口】 ===
        # 获取样本权重，如果没有这一列，默认为 1.0
        if "causal_weight" in self.df.columns:
            weight = torch.tensor(
                self.df.iloc[idx]["causal_weight"], dtype=torch.float32
            )
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)

        return image, meta, target, weight


def get_preprocessed_df(csv_path):
    """
    负责读取 CSV 并做简单的预处理（缺失值填充、One-Hot编码）
    """
    df = pd.read_csv(csv_path)

    # 1. 填充缺失值
    # 'anatom_site_general_challenge' 是最重要的分类特征之一
    df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].fillna(
        "unknown"
    )
    df["sex"] = df["sex"].fillna("unknown")
    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].mean())  # 年龄填平均值

    # 2. 归一化年龄 (否则年龄值太大，如 60，会让网络训练不稳)
    df["age_approx"] = df["age_approx"] / 100.0

    # 3. One-Hot 编码 (将文本转数字)
    # 将 'sex' 和 'anatom_site...' 拆分成多列 0/1
    df = pd.get_dummies(
        df,
        columns=["sex", "anatom_site_general_challenge"],
        prefix=["sex", "site"],
        dtype=float,
    )

    return df

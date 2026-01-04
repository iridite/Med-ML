import timm
import torch
import torch.nn as nn


class CausalFusionModel(nn.Module):
    def __init__(self, meta_features_dim, out_dim=1, pretrained=True):
        """
        参数:
            meta_features_dim: 表格数据有多少列 (dataset自动计算)
            out_dim: 输出维度 (1表示二分类)
        """
        super().__init__()

        # 1. 图像分支: EfficientNet-B0
        # num_classes=0 表示去掉原本的 1000类分类头，只保留特征层
        # in_chans=3 表示输入是 RGB 图像
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0, in_chans=3
        )
        self.img_feature_dim = self.backbone.num_features  # EfficientNet-B0 默认是 1280

        # 2. 表格分支: MLP
        self.meta_net = nn.Sequential(
            nn.Linear(meta_features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.meta_feature_dim = 128

        # 3. 融合后的总维度
        total_dim = self.img_feature_dim + self.meta_feature_dim

        # 4. 主分类头
        self.classifier = nn.Linear(total_dim, out_dim)

        # === 【因果推断预留接口】 ===
        # 这是一个辅助头，用于对抗训练或者预测混淆变量
        # 未来如果你要进行去偏见 (De-biasing)，就在这里做文章
        # 比如：让 features 无法预测 "有没有尺子" 或 "哪个医院"
        self.auxiliary_head = nn.Linear(total_dim, 5)

    def forward(self, image, meta, return_features=False):
        # 1. 提取图像特征 [Batch, 1280]
        img_feat = self.backbone(image)

        # 2. 提取表格特征 [Batch, 128]
        meta_feat = self.meta_net(meta)

        # 3. 特征融合 [Batch, 1408]
        # 这是整个模型中最关键的 Representation Layer
        features = torch.cat((img_feat, meta_feat), dim=1)

        # 4. 预测 [Batch, 1]
        logits = self.classifier(features)

        # 支持因果Loss需要拿到中间层特征
        if return_features:
            return logits, features

        return logits

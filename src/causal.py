import torch


def compute_sample_weights(df):
    """
    【预留】未来在这里计算倾向性得分 (Propensity Score)
    目前暂时返回全 1 的权重。
    """
    weights = torch.ones(len(df))
    return weights

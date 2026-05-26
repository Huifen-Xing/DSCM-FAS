import torch.nn.functional as F
import torch.nn as nn
import torch


class ContrastLoss(nn.Module):

    def __init__(self):
        super(ContrastLoss, self).__init__()
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        # 转换 contrast_label 为浮点型
        contrast_label = contrast_label.float()
        # 将 anchor_fea 从计算图中分离，以避免反向传播时更新
        anchor_fea = anchor_fea.detach()
        # 计算 anchor_fea 和 reassembly_fea 之间的余弦相似度
        loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
        # 根据 contrast_label 调整损失值
        loss = loss * contrast_label
        # 返回平均损失
        return loss.mean()


def AAMS(logits, spoof_label, type_label, num_classes):
    s = 30
    log_terms = []
    # type_label 乘以 2，用于将标签转换为双类标签
    type_label = type_label * 2
    # 将 spoof_label 从 GPU 传回 CPU，并将其元素翻倍
    spoof_label = list(spoof_label.data.cpu().numpy()) * 2
    for i, logit in enumerate(logits):
        cls = type_label[i]
        # 设置 live 和 spoof 样本的边距 m
        if spoof_label[i] == 1:  # live
            m = 0.4
        else:  # spoof
            m = 0.1
        # 生成类别的 one-hot 编码，并将其转移到与 logits 相同的设备上
        pos_mask = F.one_hot(torch.Tensor([cls]).long(), num_classes=num_classes)[0].to(logits.device)
        neg_mask = 1 - pos_mask

        # 调整 logits，减去正类边距并乘以尺度因子 s
        logit_am = s * (logit - pos_mask * m)
        logit_max = torch.max(logit_am)
        logit_am = logit_am - logit_max.detach()

        # 计算正样本项
        pos_term = (logit_am * pos_mask).sum() / (pos_mask.sum() + 1e-10)
        # 计算负样本项
        neg_term = (torch.exp(logit_am)).sum()

        # 计算对数概率差异
        log_term = pos_term - torch.log(neg_term + 1e-15)
        log_terms.append(log_term)

    # 计算损失为所有对数概率的平均值的负值
    loss = -sum(log_terms) / len(log_terms)
    return loss


def feat_sim_loss(feat1, feat2):
    # 计算 feat1 和 feat2 之间的欧氏距离，并返回平均值
    return torch.norm(feat1 - feat2, dim=1).mean()


def supcon_loss(features, labels=None, mask=None, temperature=0.1):
    base_temperature = 0.07
    """计算模型损失。如果 `labels` 和 `mask` 都未定义，
    则退化为 SimCLR 无监督损失:
    https://arxiv.org/pdf/2002.05709.pdf
    参数:
        features: 隐藏向量，形状为 [bsz, n_views, ...]。
        labels: 真实标签，形状为 [bsz]。
        mask: 对比掩码，形状为 [bsz, bsz]，mask_{i,j}=1 如果样本 j
            与样本 i 同类。可以是非对称的。
    返回:
        损失标量。
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    # 确保 features 至少有三个维度
    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        # 将 features 重塑为 3D 张量
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        # 如果未定义标签和掩码，则使用单位矩阵作为掩码
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        # 根据标签生成掩码
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        # 将现有掩码转换为浮点型
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # 计算 logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # 为了数值稳定性，减去最大值
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # 平铺掩码
    mask = mask.repeat(anchor_count, contrast_count)
    # 掩盖自身对比的情况
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # 计算 log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # 计算正样本对数似然的平均值
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # 计算损失
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def simclr_loss(features):
    # 创建标签，每两个连续的特征视为一对
    labels = torch.cat([torch.arange(len(features) // 2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    labels = labels.to(device)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)

    # 从标签和相似度矩阵中去掉主对角线
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # 选择正样本
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # 选择负样本
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # 合并正负样本，并对正样本进行标记
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # 计算损失
    logits = logits / 0.7
    criterion = torch.nn.CrossEntropyLoss().to(device)
    return criterion(logits, labels)





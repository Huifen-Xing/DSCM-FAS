import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSpectralConsistencyLoss(nn.Module):
    def __init__(self, in_channels,proj_dim=64):
        """
        Graph Spectral Consistency Loss 模块
        Args:
            in_channels: 输入特征通道数 (Layer4 输出 C)
            proj_dim: 投影后的 patch embedding 维度
            tau: 可调：0.05 ~ 0.2
        """
        super().__init__()
        self.proj_dim = proj_dim
        # 可训练投影层
        self.proj_layer = nn.Conv2d(in_channels, proj_dim, kernel_size=1)

    def forward(self, fine_feat):
        """
        fine_feat: [B, C, H, W] Layer4 输出
        return: scalar loss
        """
        B, C, H, W = fine_feat.shape
        device = fine_feat.device
        N = H * W

        # --- 1. 投影到 proj_dim ---
        proj_feat = self.proj_layer(fine_feat)  # [B, proj_dim, H, W]

        # --- 2. 展平 patch 到列向量 ---
        patch_feat = proj_feat.view(B, self.proj_dim, N)  # [B, proj_dim, N]
        patch_feat = F.normalize(patch_feat, p=2, dim=1)  # L2 归一化列向量

        # --- 3. 计算 patch 相似性矩阵 V ---
        V = torch.bmm(patch_feat.transpose(1,2), patch_feat)
        V = torch.exp(V / 0.07)
        #V = V / (V.sum(dim=-1, keepdim=True) + 1e-6)
        '''
        # --- 4. 计算 V 的均方差损失 ---
        V_mean = V.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        loss_var = ((V - V_mean) ** 2).mean()

        return loss_var
        '''
        #####################################
        
        # --- 4. 构造度矩阵 D 和拉普拉斯 L ---
        #D = torch.sum(V, dim=2)
        #D = torch.diag_embed(D)
        #L = D - V  # unnormalized Laplacian
        deg = V.sum(dim=-1)                       # [B, N]
        D_inv_sqrt = torch.diag_embed((deg + 1e-6).pow(-0.5))

        I = torch.eye(N, device=V.device).unsqueeze(0)
        L_sym = I - D_inv_sqrt @ V @ D_inv_sqrt    # [B, N, N]

        # --- 5. Graph Spectral Loss proxy ---
        # 用 Trace(L) / N 近似 λ2
        # 优点: GPU安全, batch-friendly, 高效
        #traces = L.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B]
        #loss = traces.mean() / N
        loss = torch.einsum('bni,bij,bnj->b', patch_feat, L_sym,patch_feat).mean()/10
        
        return loss
        '''
        B, N, _ = V.shape
        mask = ~torch.eye(N, device=V.device).bool()
        V_off = V[:, mask].view(B, N, N-1)

        V_mean = V_off.mean(dim=(1,2), keepdim=True)
        loss_var = ((V_off - V_mean)**2).mean()
        return loss_var
        '''
def graph_consistency_loss(
    feat_o,
    feat_aug,
    global_latent,
    k=5,
    top_M=10
):
    B, D = feat_o.shape

    # --- 2. 选择 top-M neighbor (在全局 latent 中) ---
    # 计算 feat_o 与 global_latent 的相似度
    sim = torch.matmul(feat_o, global_latent.T)  # [B, N]
    topk_idx = torch.topk(sim, k=top_M, dim=1).indices  # [B, top_M]

    # --- 3. 构建 H 矩阵 (M+1 个矩阵拼接) ---
    neighbor_feats = global_latent[topk_idx.view(-1)]  # [B*top_M, D]
    H = torch.cat([feat_aug, neighbor_feats.view(B*top_M, D)], dim=0)  # [(1+top_M)*B, D]
    
    # --- 4. 构建 anchors ---
    anchors = [feat_aug]  # 可以包含 feat_o, aug_feat 等 k+1 个正样本
    for r in range(k):
        anchors.append(global_latent[topk_idx[:, r]])

    # --- 5. 计算 similarity subgraph A_hat ---
    A_list = []
    for z_r in anchors:
        A_hat = torch.matmul(z_r, H.T) # [B, (M+1)*B]
        A_list.append(A_hat)

    # --- 6. 计算 Graph Consistency Loss ---
    loss = 0.
    cnt = 0
    for i in range(len(A_list)):
        for j in range(i+1, len(A_list)):
            loss += ((A_list[i] - A_list[j]) ** 2).mean()*10
            cnt += 1

    return loss / cnt
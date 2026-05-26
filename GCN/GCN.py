import torch
import torch.nn as nn
import torch.nn.functional as F

def build_topk_adj(feat1,k=15, symmetric=True):
    """
    Implements:
        A_tilde = A + I
        D_tilde = D + I
        A_hat   = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
    """

    B = feat1.size(0)
    device = feat1.device

    # ---------- 1. 相似性 ----------
    sim = torch.matmul(feat1, feat1.t())   # [B, B]
    #sim = torch.exp(sim / 0.1)

    # ---------- 2. Top-k 稀疏化 ----------
    topk_val, topk_idx = torch.topk(sim, k=k, dim=1)
    A = torch.zeros_like(sim)
    A.scatter_(1, topk_idx, topk_val)

    # ---------- 3. 对称化 ----------
    if symmetric:
        A = torch.maximum(A, A.t())

    # ---------- 4. Ã = A + I ----------
    I = torch.eye(B, device=device)
    A_tilde = A + I

    # ---------- 5. D̃ = D + I ----------
    deg_tilde = A_tilde.sum(dim=1)              # D (不含自环)
    deg_tilde = deg_tilde + 1.0            # D + I

    # ---------- 6. D̃^{-1/2} ----------
    #D_inv_sqrt = torch.diag(torch.pow(deg_tilde, -0.5))
    eps = 1e-6
    D_inv_sqrt = torch.diag((deg_tilde + eps).pow(-0.5))

    # ---------- 7. A_hat ----------
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    #A_hat = D @ A_tilde @ D

    return A_hat

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        #output = torch.spmm(adj, support)
        output = adj @ support
        if self.use_bias:
            return output + self.bias
        else:
            return output
        
class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=2,dropout = 0.2):
        """
        二层图卷积网络
        Args:
            in_dim: 输入特征维度 D
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
        """
        super().__init__()
        self.gcn1 = GraphConvolution(in_dim, hidden_dim, bias=False)
        self.gcn2 = GraphConvolution(hidden_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_hat):
        """
        X: [B, D] 原始节点特征
        A_hat: [B, B] 归一化邻接矩阵
        Returns:
            logits: [B, num_classes] 每个节点的分类预测
        """
        # 第一层 GCN + ReLU
        H = F.relu(self.gcn1(X,A_hat))  # @ 是矩阵乘法
        #H = self.dropout(H)
        # 第二层 GCN
        logits = self.gcn2(H, A_hat)
        return H,logits
    
    
def build_topk_adj_gat(feat, k=10, symmetric=True):
    """
    返回 GAT 使用的邻接矩阵（0/1 或带权）
    """
    B = feat.size(0)
    device = feat.device

    # 相似度
    sim = torch.matmul(feat, feat.t())  # [B, B]

    # Top-k
    _, topk_idx = torch.topk(sim, k=k, dim=1)
    adj = torch.zeros_like(sim)
    adj.scatter_(1, topk_idx, 1.0)  # 只保留连接关系

    if symmetric:
        adj = torch.maximum(adj, adj.t())

    # 加自环（GAT 通常需要）
    adj = adj + torch.eye(B, device=device)

    return adj


class GraphAttentionLayer(nn.Module):
    """
    单层图注意力机制（GAT Layer）
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat  # True 表示中间层，False 表示输出层

        # 权重矩阵 W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力机制参数 a
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h: 节点特征矩阵 [N, in_features]
        adj: 邻接矩阵 [N, N] (0/1)
        """
        # 输入验证
        if h.size(0) != adj.size(0):
            raise ValueError("节点数与邻接矩阵维度不匹配")

        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size(0)

        # 计算注意力系数 e_ij
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1),
                             Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 只对邻居节点计算注意力
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # softmax 归一化
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # 聚合邻居信息
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime  # 输出层不加激活

class GAN(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, dropout=0.6, alpha=0.2, n_heads=8):
        super(GAN, self).__init__()
        self.dropout = dropout

        # 多头注意力机制
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_feat, n_hidden, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(n_heads)
        ])

        # 输出层（多头拼接 -> 单头输出）
        self.out_att = GraphAttentionLayer(n_hidden * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        #z = F.log_softmax(x, dim=1)
        return x
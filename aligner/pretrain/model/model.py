import copy

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from tqdm import tqdm
from .layers import SAGEConv
from torch_geometric.nn.conv import GATConv, GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusion_original(nn.Module):
    def __init__(self,
                 gph_dim, img_dim=None, attr_dim=None, rel_dim=None,
                 hidden_dim=300):

        super().__init__()
        self.hidden_dim = hidden_dim

        # 投影层
        self.proj_gph = nn.Linear(gph_dim, hidden_dim)
        self.proj_img = nn.Linear(img_dim, hidden_dim) if img_dim is not None else None
        self.proj_attr = nn.Linear(attr_dim, hidden_dim) if attr_dim is not None else None
        self.proj_rel = nn.Linear(rel_dim, hidden_dim) if rel_dim is not None else None

        # 模态注意力融合权重
        self.fusion_proj = nn.Linear(hidden_dim, 1)

    def forward(self, gph_emb, img_emb=None, attr_emb=None, rel_emb=None):

        gph_emb = self.proj_gph(gph_emb)
        all_embs = [gph_emb]
        if img_emb is not None and self.proj_img is not None:
            img_emb = self.proj_img(img_emb)
            all_embs.append(img_emb)
        if attr_emb is not None and self.proj_attr is not None:
            attr_emb = self.proj_attr(attr_emb)
            all_embs.append(attr_emb)
        if rel_emb is not None and self.proj_rel is not None:
            rel_emb = self.proj_rel(rel_emb)
            all_embs.append(rel_emb)

        # 计算每个模态的注意力分数
        att_list = [self.fusion_proj(e) for e in all_embs]  # 每个 [N, 1]
        att_weights = torch.softmax(torch.cat(att_list, dim=-1), dim=-1)  # [N, M]

        # 加权融合
        fused_emb = sum(att_weights[:, i].unsqueeze(-1) * all_embs[i]
                        for i in range(len(all_embs)))

        return F.normalize(fused_emb, p=2, dim=-1), img_emb, attr_emb, rel_emb


class MultiModalFusion_attention(nn.Module):


    def __init__(self,
                 gph_dim, img_dim=None, attr_dim=None, rel_dim=None,
                 hidden_dim=300,
                 num_queries=4,  # K 查询 token 数 (可调)
                 num_heads=4):  # 多头注意力头数 (可调)
        super().__init__()

        self.hidden_dim = hidden_dim

        self.proj_gph = nn.Linear(gph_dim, hidden_dim)
        self.proj_img = nn.Linear(img_dim, hidden_dim) if img_dim is not None else None
        self.proj_attr = nn.Linear(attr_dim, hidden_dim) if attr_dim is not None else None
        self.proj_rel = nn.Linear(rel_dim, hidden_dim) if rel_dim is not None else None

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.query = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, gph_emb, img_emb=None, attr_emb=None, rel_emb=None):

        tokens = [self.proj_gph(gph_emb).unsqueeze(1)]  # [N, 1, H]
        if img_emb is not None and self.proj_img is not None:
            tokens.append(self.proj_img(img_emb).unsqueeze(1))
        if attr_emb is not None and self.proj_attr is not None:
            tokens.append(self.proj_attr(attr_emb).unsqueeze(1))
        if rel_emb is not None and self.proj_rel is not None:
            tokens.append(self.proj_rel(rel_emb).unsqueeze(1))

        kv = torch.cat(tokens, dim=1)  # [N, M, H], M=有效模态数

        kv, _ = self.self_attn(kv, kv, kv)  # [N, M, H]

        B = kv.size(0)
        q = self.query.expand(B, -1, -1)  # [N, K, H]
        fused_tokens, _ = self.cross_attn(q, kv, kv)  # [N, K, H]

        fused_emb = fused_tokens.mean(dim=1)  # [N, H]

        return F.normalize(fused_emb, p=2, dim=-1), img_emb, attr_emb, rel_emb


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, num_proj_hidden, activation, dropout,
                 graph_pooling='sum', edge_dim=None, gnn_type='sage'):
        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = hidden_channels
        self.n_classes = out_channels
        self.convs = torch.nn.ModuleList()
        if gnn_type == 'sage':
            gnn_conv = SAGEConv
        elif gnn_type == "gat":
            gnn_conv = GATConv
        elif gnn_type == 'gcn':
            gnn_conv = GCNConv

        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        # non-linear layer for contrastive loss
        self.fc1 = torch.nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, out_channels)

        self.dropout = dropout
        self.activation = activation

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index, edge_attr=edge_attr)
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is not None:
            x = self.pool(x, batch)

        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class MultiModalGraphSAGE(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 n_layers,
                 num_proj_hidden,
                 activation,
                 dropout,
                 graph_pooling='sum',
                 edge_dim=None,
                 gnn_type='sage',
                 # === 多模态参数 ===
                 img_dim=None,
                 attr_dim=None,
                 rel_dim=None,
                 fusion_hidden_dim=None,
                 fusion=None
                 ):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

        if gnn_type == 'sage':
            gnn_conv = SAGEConv
        elif gnn_type == 'gat':
            gnn_conv = GATConv
        elif gnn_type == 'gcn':
            gnn_conv = GCNConv
        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")

        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for _ in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))
        elif graph_pooling.startswith("set2set"):
            set2set_iter = int(graph_pooling[-1])
            self.pool = Set2Set(out_channels, set2set_iter)
        else:
            raise ValueError(f"Unknown graph_pooling: {graph_pooling}")

        self.fc1 = nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, out_channels)

        if fusion == 'original':
            print("原始跨模态融合")
            self.fusion = MultiModalFusion_original(
                gph_dim=out_channels,
                img_dim=img_dim,
                attr_dim=attr_dim,
                rel_dim=rel_dim,
                hidden_dim=fusion_hidden_dim
            )
        if fusion == 'attention':
            print("注意力跨模态融合")
            self.fusion = MultiModalFusion_attention(
                gph_dim=out_channels,
                img_dim=img_dim,
                attr_dim=attr_dim,
                rel_dim=rel_dim,
                hidden_dim=fusion_hidden_dim
            )

    def forward(self, x, edge_index, img_emb=None, attr_emb=None, rel_emb=None, batch=None):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is not None:
            x = self.pool(x, batch)

        fused_emb, img_emb, attr_emb, rel_emb = self.fusion(
            gph_emb=x,
            img_emb=img_emb,
            attr_emb=attr_emb,
            rel_emb=rel_emb
        )

        return fused_emb, x, img_emb, attr_emb, rel_emb

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

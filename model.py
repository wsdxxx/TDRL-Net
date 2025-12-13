import copy
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GATv2Conv
import torch.nn.functional as F

class MetaPathGATv2Layer(nn.Module):
    def __init__(self, meta_paths, in_feat, out_feat, num_heads):
        super().__init__()
        self.meta_paths = meta_paths
        self.residual_drop = nn.Dropout(0.5)

        self.gat_layers = nn.ModuleDict({
            f'mp_{i}': GATv2Conv(
                in_feat, out_feat, num_heads,
                feat_drop=0.3, attn_drop=0.3,
                residual=True, activation=nn.ELU(),
                allow_zero_in_degree=True
            ) for i, _ in enumerate(meta_paths)
        })

        self.fnn_fuse = nn.Sequential(
            nn.Linear(out_feat * num_heads * len(meta_paths), 256),
            nn.ReLU(),
            nn.Linear(256, out_feat * num_heads)
        )

        self.gate = nn.Sequential(
            nn.Linear(out_feat * num_heads, 1),
            nn.Sigmoid()
        )

    def forward(self, g, h_dict):


        metapath_graphs = {}
        for i, path in enumerate(self.meta_paths):
            metapath_subgraph = dgl.metapath_reachable_graph(g, path)
            metapath_graphs[f'mp_{i}'] = dgl.add_self_loop(metapath_subgraph)

        semantic_embeds = []
        for i, mp_name in enumerate(metapath_graphs):
            mp_g = metapath_graphs[mp_name]
            source_node_type = self.meta_paths[i][0][0]
            h = h_dict[source_node_type]
            h = self.gat_layers[mp_name](mp_g, h).flatten(1)
            semantic_embeds.append(h)

        stacked_embeds = torch.cat(semantic_embeds, dim=1)
        fused_output = self.fnn_fuse(stacked_embeds)

        base_output = torch.mean(torch.stack(semantic_embeds, dim=0), dim=0)
        gate_score = self.gate(base_output)
        final_output = gate_score * fused_output + (1 - gate_score) * base_output

        return final_output

class discriminator(nn.Module):
    def __init__(self, itemCount, feat_shape, out_feat_shape):
        super(discriminator, self).__init__()
        self.itemCount = itemCount
        self.feat_shape = feat_shape
        self.out_feat_shape = out_feat_shape

        self.net = nn.Sequential(
            nn.Linear(self.itemCount + self.out_feat_shape, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, Adj, embedding):
        x = torch.cat((Adj, embedding), 1)
        return self.net(x)


class generator(nn.Module):
    def __init__(self, itemCount, feat_shape, out_feat_shape, meta_paths, lambda_contrastive=0.4, lambda_recon=0.3, lambda_edge=0.3):
        super(generator, self).__init__()
        self.itemCount = itemCount
        self.feat_shape = feat_shape
        self.out_feat_shape = out_feat_shape
        self.meta_paths = meta_paths
        self.lambda_contrastive = lambda_contrastive
        self.lambda_recon = lambda_recon
        self.lambda_edge = lambda_edge

        self.mpgat_layers = nn.ModuleList([
            MetaPathGATv2Layer(
                meta_paths=meta_paths,
                in_feat=feat_shape if i == 0 else 256,
                out_feat=128,
                num_heads=2
            ) for i in range(2)
        ])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(2)])
        self.residual_proj = nn.ModuleList([
            nn.Linear(256, 256) if i != 0 else nn.Identity()
            for i in range(2)
        ])

        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_feat_shape),
            nn.LayerNorm(out_feat_shape)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.itemCount + self.out_feat_shape, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, itemCount),
            nn.Sigmoid()
        )

        self.feature_reconstruct_head = nn.Sequential(
            nn.Linear(out_feat_shape, 256),
            nn.ReLU(),
            nn.Linear(256, feat_shape)
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(self.out_feat_shape + self.feat_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, g, h, Adj, size, leftIndex):
        device = next(self.parameters()).device
        h_dict = copy.deepcopy(h)
        h_dict['LncRNA'] = h_dict['LncRNA'].to(device)
        h_residual = h_dict['LncRNA']

        for i, layer in enumerate(self.mpgat_layers):
            h_mp = layer(g, h_dict)
            h_mp = F.dropout(h_mp, p=0.4 + i * 0.1, training=self.training)
            if h_mp.shape == h_residual.shape:
                h_mp += self.residual_proj[i](h_residual)
            h_mp = self.bn_layers[i](h_mp)
            h_dict['LncRNA'] = F.elu(h_mp) if i != len(self.mpgat_layers) - 1 else h_mp
            h_residual = h_dict['LncRNA']

        projected_feat = self.projection(h_dict['LncRNA'])
        lncrna_feat = F.layer_norm(projected_feat, [projected_feat.size(1)])
        fake_embedding = lncrna_feat[leftIndex:leftIndex + size].clone()

        M = torch.cat([Adj.to(device), fake_embedding], dim=1)
        output = self.mlp(M)

        contrastive_loss = self.compute_contrastive_loss(
            fake_embedding, h_dict['LncRNA'][leftIndex:leftIndex + size])

        recon_input = projected_feat[leftIndex:leftIndex + size]
        recon_target = h['LncRNA'][leftIndex:leftIndex + size].to(device)
        recon_pred = self.feature_reconstruct_head(recon_input)
        recon_loss = F.mse_loss(recon_pred, recon_target)

        edge_preds, edge_labels = self.compute_edge_pred_loss(
            fake_embedding, h_dict['Disease'], Adj
        )
        edge_loss = F.binary_cross_entropy(edge_preds, edge_labels)

        total_aux_loss = (
            self.lambda_contrastive * contrastive_loss +
            self.lambda_recon * recon_loss +
            self.lambda_edge * edge_loss
        )

        return fake_embedding, output, total_aux_loss

    def compute_contrastive_loss(self, fake_embedding, real_embedding, temperature=0.6):
        fake_norm = F.normalize(fake_embedding, dim=1)
        real_norm = F.normalize(real_embedding, dim=1)
        logits = torch.matmul(fake_norm, real_norm.T) / temperature
        labels = torch.arange(logits.size(0)).to(logits.device)
        return F.cross_entropy(logits, labels)

    def compute_edge_pred_loss(self, lnc_embeds, dis_embeds, true_adj_row):
        bsz, dis_num = true_adj_row.shape
        device = lnc_embeds.device
        lnc_exp = lnc_embeds.unsqueeze(1).repeat(1, dis_num, 1)
        dis_exp = dis_embeds.unsqueeze(0).repeat(bsz, 1, 1)
        pairwise = torch.cat([lnc_exp, dis_exp], dim=-1)
        edge_pred = self.edge_predictor(pairwise).squeeze(-1)
        return edge_pred.view(-1), true_adj_row.view(-1).to(device)



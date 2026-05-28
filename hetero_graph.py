#构建并保存异构图
import os
import dgl
import torch
from dgl import save_graphs, load_graphs
import torch.nn.functional as F

def get_hetero_graph(input_net, true_input_net, lncrna_mirna, mirna_disease, lncrna_feat, mirna_feat, disease_feat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 新增元路径定义
    meta_paths = [
        [("LncRNA", "LvsM", "MiRNA"), ("MiRNA", "MvsL", "LncRNA")],  # L-M-L
        [("LncRNA", "LvsD", "Disease"), ("Disease", "DvsL", "LncRNA")],  # L-D-L
        [("LncRNA", "LvsM", "MiRNA"), ("MiRNA", "MvsD", "Disease"),
         ("Disease", "DvsM", "MiRNA"), ("MiRNA", "MvsL", "LncRNA")],  # L-M-D-M-L
        [("LncRNA", "LvsD", "Disease"), ("Disease", "DvsM", "MiRNA"), ("MiRNA", "MvsL", "LncRNA")]  # L-D-M-L
    ]

    # 新增显式指定节点数量为特征矩阵的行数
    num_nodes_dict = {
        'LncRNA': lncrna_feat.shape[0],
        'MiRNA': mirna_feat.shape[0],
        'Disease': disease_feat.shape[0]
    }

    Noise_LMDN_data = {
        ('LncRNA', 'LvsD', 'Disease'): (torch.tensor(input_net[0]), torch.tensor(input_net[1])),
        ('Disease', 'DvsL', 'LncRNA'): (torch.tensor(input_net[1]), torch.tensor(input_net[0])),
        ('LncRNA', 'LvsM', 'MiRNA'): (torch.tensor(lncrna_mirna[0]), torch.tensor(lncrna_mirna[1])),
        ('MiRNA', 'MvsL', 'LncRNA'): (torch.tensor(lncrna_mirna[1]), torch.tensor(lncrna_mirna[0])),
        ('MiRNA', 'MvsD', 'Disease'): (torch.tensor(mirna_disease[0]), torch.tensor(mirna_disease[1])),
        ('Disease', 'DvsM', 'MiRNA'): (torch.tensor(mirna_disease[1]), torch.tensor(mirna_disease[0]))
    }
    # Noise_LMDN = dgl.heterograph(Noise_LMDN_data)
    Noise_LMDN = dgl.heterograph(Noise_LMDN_data, num_nodes_dict=num_nodes_dict).to(device)  # 关键修改

    Noise_LMDN.nodes['LncRNA'].data['feat'] = F.layer_norm(
        lncrna_feat, [lncrna_feat.size(1)]
    )  # 新增归一化
    # Noise_LMDN.nodes['MiRNA'].data['feat'] = F.layer_norm(
        # mirna_feat, [mirna_feat.size(1)]
    # )  # 新增归一化
    # Noise_LMDN.nodes['Disease'].data['feat'] = F.layer_norm(
        # disease_feat, [disease_feat.size(1)]
    # )  # 新增归一化
    Noise_LMDN.nodes['MiRNA'].data['feat'] = mirna_feat
    Noise_LMDN.nodes['Disease'].data['feat'] = disease_feat
    Noise_LMDN_h = {'LncRNA': Noise_LMDN.nodes['LncRNA'].data['feat'],
                    'MiRNA': Noise_LMDN.nodes['MiRNA'].data['feat'],
                    'Disease': Noise_LMDN.nodes['Disease'].data['feat']}


    True_LMDN_data = {
        ('LncRNA', 'LvsD', 'Disease'): (torch.tensor(true_input_net[0]), torch.tensor(true_input_net[1])),
        ('Disease', 'DvsL', 'LncRNA'): (torch.tensor(true_input_net[1]), torch.tensor(true_input_net[0])),
        ('LncRNA', 'LvsM', 'MiRNA'): (torch.tensor(lncrna_mirna[0]), torch.tensor(lncrna_mirna[1])),
        ('MiRNA', 'MvsL', 'LncRNA'): (torch.tensor(lncrna_mirna[1]), torch.tensor(lncrna_mirna[0])),
        ('MiRNA', 'MvsD', 'Disease'): (torch.tensor(mirna_disease[0]), torch.tensor(mirna_disease[1])),
        ('Disease', 'DvsM', 'MiRNA'): (torch.tensor(mirna_disease[1]), torch.tensor(mirna_disease[0]))
    }
    # True_LMDN = dgl.heterograph(True_LMDN_data)
    True_LMDN = dgl.heterograph(True_LMDN_data, num_nodes_dict=num_nodes_dict).to(device)  # 关键修改
    True_LMDN.nodes['LncRNA'].data['feat'] = lncrna_feat
    True_LMDN.nodes['MiRNA'].data['feat'] = mirna_feat
    True_LMDN.nodes['Disease'].data['feat'] = disease_feat
    True_LMDN_h = {'LncRNA': True_LMDN.nodes['LncRNA'].data['feat'],
                   'MiRNA': True_LMDN.nodes['MiRNA'].data['feat'],
                   'Disease': True_LMDN.nodes['Disease'].data['feat']}

    # Save hetero graph
    # save_path = './save_hetero_graph'
    # Noise_LMDN_mode = 'Noise_LMDN'
    # Noise_LMDN_path = os.path.join(save_path, Noise_LMDN_mode + '.bin')
    # save_graphs(Noise_LMDN_path, Noise_LMDN ,)
    # True_LMDN_mode = 'True_LMDN'
    # True_LMDN_path = os.path.join(save_path, True_LMDN_mode + '.bin')
    # save_graphs(True_LMDN_path, True_LMDN,)
    # Load hetero graph
    # Noise_LMDN, _ = load_graphs(Noise_LMDN_path)
    # True_LMDN, _ = load_graphs(True_LMDN_path)

    return Noise_LMDN, Noise_LMDN_h, True_LMDN, True_LMDN_h, meta_paths# 添加返回值
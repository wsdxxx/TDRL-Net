import torch
import train
import random
import numpy as np
import model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import f1_score, recall_score
from Data.load_Dataset1 import dataset1
from Data.load_Dataset2 import dataset2
from Data.load_Dataset3 import dataset3
from Data.load_Dataset4 import dataset4
from hetero_graph import get_hetero_graph

import warnings

warnings.filterwarnings("ignore")


# 设置随机种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 自动生成一个随机种子并记录
    current_seed = 52262 #D1 D2
    # current_seed = 17927 #D3
    # current_seed = random.randint(1, 99999)

    print(f"🌱 当前随机种子为: {current_seed}")

    # 设置当前种子
    set_random_seed(current_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 150  #
    pro_ZR = 30#D1:30   D3:20
    pro_PM = 70  #D1:70  D3:80
    alpha = 0.1 #D1:0.1  D3:0.2
    #0.05 Mean AUC: 0.8906, Mean AUPR: 0.9048, Mean ACC: 0.7417, Mean MCC: 0.5516, Mean Recall: 0.5033, Mean F1: 0.6592, Mean Precision: 0.9684
    #0.1 Mean AUC: 0.8913, Mean AUPR: 0.9039, Mean ACC: 0.7391, Mean MCC: 0.5509, Mean Recall: 0.4919, Mean F1: 0.6524, Mean Precision: 0.9741
    #0.2 Mean AUC: 0.8914, Mean AUPR: 0.9065, Mean ACC: 0.7359, Mean MCC: 0.5465, Mean Recall: 0.4844, Mean F1: 0.6463, Mean Precision: 0.9755
    #0.3 Mean AUC: 0.8892, Mean AUPR: 0.9067, Mean ACC: 0.7385, Mean MCC: 0.5526, Mean Recall: 0.4882, Mean F1: 0.6498, Mean Precision: 0.9732
    #10:90  20:80 30:70
    feat_shape = 64
    out_feat = 256

    G_step = 5
    D_step = 2
    batchSize = 32

    G_PATH = './weights/G.pth'
    D_PATH = './weights/D.pth'

    AUC = []
    AUPR = []
    ACC, MCC,RECALL,F1,PRECISION = [], [],[], [],[]
    # 加载数据集
    lncrna_num, mirna_num, disease_num, lncrna_disease, mirna_disease, lncrna_mirna = dataset1()

    # lncrna_num, mirna_num, disease_num, lncrna_disease, mirna_disease, lncrna_mirna, lncRNA_name = dataset2()
    # lncrna_num, mirna_num, disease_num, lncrna_disease, mirna_disease, lncrna_mirna = dataset3()
    # lncrna_num, mirna_num, disease_num, lncrna_disease, mirna_disease, lncrna_mirna = dataset4()

    # lncrna_feat = get_k_mer_feat.get_feature(lncRNA_name)
    lncrna_feat = torch.rand(lncrna_num, feat_shape).to(device)
    print(f"lncrna_num: {lncrna_num}, lncrna_feat shape: {lncrna_feat.shape}")

    mirna_feat = torch.rand(mirna_num, feat_shape).to(device)
    disease_feat = torch.rand(disease_num, feat_shape).to(device)

    print("\nData Details:")
    print("lncRNA:{}  miRNA:{}  Disease:{}".format(lncrna_num, mirna_num, disease_num))
    print("lncRNA-Disease:{}  miRNA-Disease:{}  lncRNA-miRNA:{}".format(len(lncrna_disease[0]), len(mirna_disease[0]),
                                                                        len(lncrna_mirna[0])))
    print("Sparsity of lncRNA-Disease associated data: {:.6f}\n".format(
        len(lncrna_disease[0]) / (lncrna_num * disease_num)))

    edge_data = []
    for i in range(len(lncrna_disease[0])):
        x = []
        x.append(lncrna_disease[0][i])
        x.append(lncrna_disease[1][i])
        edge_data.append(x)
    edge_data = np.array(edge_data)

    negativeSample_edge = []
    for i in range(len(edge_data)):
        row = random.randint(0, lncrna_num - 1)
        col = random.randint(0, disease_num - 1)
        while ([row, col] in edge_data.tolist() or [row, col] in negativeSample_edge):
            row = random.randint(0, lncrna_num - 1)
            col = random.randint(0, disease_num - 1)
        negativeSample_edge.append([row, col])

    import json
    import os
    from sklearn.metrics import roc_auc_score, average_precision_score

    if not os.path.exists("results"):
        os.makedirs("results")

    fold_metrics = {}

    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(edge_data):
        train_negative = random.sample(negativeSample_edge, int(len(train_index)))
        test_negative = [data_negative for data_negative in negativeSample_edge if data_negative not in train_negative]
        train_index = train_index.tolist()
        train_lncrna_20 = random.sample(train_index, int(len(train_index) * 0.25))
        train_lncrna_60 = [i for i in train_index if i not in train_lncrna_20]
        test_lncrna = test_index.tolist()

        input_net = [[], []]
        for i in train_lncrna_60:
            input_net[0].append(edge_data[i][0])
            input_net[1].append(edge_data[i][1])
        for i in range(len(train_negative)):
            input_net[0].append(train_negative[i][0])
            input_net[1].append(train_negative[i][1])

        true_input_net = [[], []]
        for i in train_index:
            true_input_net[0].append(edge_data[i][0])
            true_input_net[1].append(edge_data[i][1])

        test_input_net = [[], []]
        for i in test_lncrna:
            test_input_net[0].append(edge_data[i][0])
            test_input_net[1].append(edge_data[i][1])
        for i in range(len(test_negative)):
            test_input_net[0].append(test_negative[i][0])
            test_input_net[1].append(test_negative[i][1])

        Noise_LMDN, Noise_LMDN_h, True_LMDN, True_LMDN_h, meta_paths = get_hetero_graph(input_net, true_input_net,
                                                                                        lncrna_mirna,
                                                                                        mirna_disease, lncrna_feat,
                                                                                        mirna_feat,
                                                                                        disease_feat)
        Noise_LMDN = Noise_LMDN.to(device)
        True_LMDN = True_LMDN.to(device)
        # # 修改异构图构建调用
        # Noise_LMDN, Noise_LMDN_h, True_LMDN, True_LMDN_h, meta_paths = get_hetero_graph(
        #     input_net, true_input_net, lncrna_mirna, mirna_disease,
        #     lncrna_feat, mirna_feat, disease_feat, input_disease_disease  # 传入D-D边
        # )
        G = model.generator(disease_num, feat_shape, out_feat, meta_paths=meta_paths).to(device)
        D = model.discriminator(disease_num, feat_shape, out_feat).to(device)
        auc, aupr, acc, mcc,  recall,f1,precision,labels, preds= train.main(lncrna_num, disease_num, epochs, pro_ZR, pro_PM, alpha, batchSize,
                         input_net, true_input_net, test_input_net, test_negative,
                         Noise_LMDN, Noise_LMDN_h, True_LMDN, True_LMDN_h,
                         G, D, G_step, D_step, G_PATH, D_PATH
                         # , dynamic_params=DYNAMIC_STOP_PARAMS
                         )  # 新增参数
        AUC.append(auc)
        AUPR.append(aupr)
        ACC.append(acc)
        MCC.append(mcc)
        RECALL.append(recall)
        F1.append(f1)
        PRECISION.append(precision)
        fold_id = f'fold_{len(AUC) - 1}'
        fold_metrics[fold_id] = {
            'y_true': labels,
            'y_score': preds,
            'auc': auc,
            'aupr': aupr,
            'acc': acc,
            'mcc': mcc,
            'recall': recall,
            'f1': f1,
            'precision': precision  # ✅ 新增
        }

        print('10_fold_auc:{}'.format(AUC))
        print('10_fold_aupr:{}'.format(AUPR))
        print('10_fold_acc:{}'.format(ACC))
        print('10_fold_mcc:{}'.format(MCC))
        print('10_fold_recall:{}'.format(RECALL))
        print('10_fold_f1:{}'.format(F1))
        print('10_fold_precision:{}'.format(PRECISION))  # ✅ 新增打印

        print(
            'Mean AUC: {:.4f}, Mean AUPR: {:.4f}, Mean ACC: {:.4f}, Mean MCC: {:.4f}, '
            'Mean Recall: {:.4f}, Mean F1: {:.4f}, Mean Precision: {:.4f}\n'.format(
                sum(AUC) / len(AUC),
                sum(AUPR) / len(AUPR),
                sum(ACC) / len(ACC),
                sum(MCC) / len(MCC),
                sum(RECALL) / len(RECALL),
                sum(F1) / len(F1),
                sum(PRECISION) / len(PRECISION)  # ✅ 新增 mean precision
            ))

    fold_metrics['average'] = {
        'auc': sum(AUC) / len(AUC),
        'aupr': sum(AUPR) / len(AUPR),
        'acc': sum(ACC) / len(ACC),
        'mcc': sum(MCC) / len(MCC),
        'recall': sum(RECALL) / len(RECALL),
        'f1': sum(F1) / len(F1),
        'precision': sum(PRECISION) / len(PRECISION)  # ✅ 新增
    }

    with open('results/10fold_metricsD311.json', 'w') as f:
        json.dump(fold_metrics, f, indent=4)
    print("✅ 10折交叉验证数据已保存至 results/10fold_metricsD311.json")

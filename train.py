import copy
import torch
import random
import numpy as np
import torch.nn as nn
from get_adj import adj
from sklearn import metrics
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F  # 用于 BCE、cosine 等
from torch_optimizer import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
def select_negative_items(Data, num_pm, num_zr, disease_num):
    # 将 CUDA 张量转换为 CPU 的 NumPy 数组
    if isinstance(Data, torch.Tensor):
        data = Data.cpu().detach().numpy()
    else:
        data = np.array(Data)

    n_items_pm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)
    for i in range(data.shape[0]):
        p_items = np.where(data[i] != 0)[0]
        all_item_index = random.sample(range(data.shape[1]), disease_num)
        for j in p_items:
            if j in all_item_index:
                all_item_index.remove(j)
        random.shuffle(all_item_index)
        n_item_index_pm = all_item_index[0: num_pm]
        n_item_index_zr = all_item_index[num_pm: (num_pm + num_zr)]
        n_items_pm[i][n_item_index_pm] = 1
        n_items_zr[i][n_item_index_zr] = 1
    return n_items_pm, n_items_zr

def main(lncrna_num, disease_num, epochCount, pro_ZR, pro_PM, alpha, batchSize,
         input_net, true_input_net, test_input_net, test_negative,
         Noise_LMDN, Noise_LMDN_h, True_LMDN, True_LMDN_h,
         G, D, G_step, D_step, G_PATH, D_PATH):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_auc_labels, best_auc_preds = [], []

    best_auc = 0
    best_aupr = 0
    best_acc = 0
    best_mcc = 0
    best_recall = 0
    best_f1 = 0
    best_precision = 0
    early_stop_counter = 0
    patience = 15

    mse_loss_fn = nn.MSELoss()
    d_optimizer = torch.optim.AdamW(D.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-5) #D1:0.0005 D3:0.0004 D2:0.0002
    g_optimizer = torch.optim.AdamW(G.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler_d = CosineAnnealingLR(d_optimizer, T_max=20, eta_min=1e-6)
    scheduler_g = CosineAnnealingLR(g_optimizer, T_max=20, eta_min=1e-6)
    # # 优化器
    # g_optimizer = RAdam(G.parameters(), lr=0.0005, weight_decay=1e-5)
    # d_optimizer = RAdam(D.parameters(), lr=0.0005, weight_decay=1e-5)
    #
    # # 调度器
    # scheduler_g = CosineAnnealingLR(g_optimizer, T_max=20, eta_min=1e-6)
    # scheduler_d = ReduceLROnPlateau(d_optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    Noise_LDA = adj(lncrna_num, disease_num, input_net).to(device)
    True_LDA  = adj(lncrna_num, disease_num, true_input_net).to(device)
    Test_Adj  = adj(lncrna_num, disease_num, test_input_net).to(device)

    G.to(device); D.to(device)
    Noise_LMDN = Noise_LMDN.to(device)
    True_LMDN  = True_LMDN.to(device)
    Noise_LMDN_h = {k: v.to(device) for k, v in Noise_LMDN_h.items()}
    True_LMDN_h  = {k: v.to(device) for k, v in True_LMDN_h.items()}

    for epoch in range(epochCount):
        # —— 生成器训练 ——
        for _ in range(G_step):
            leftIndex = random.randint(0, lncrna_num - batchSize - 1)
            realData = Variable(True_LDA[leftIndex:leftIndex + batchSize]).to(device)
            noiseData = Variable(Noise_LDA[leftIndex:leftIndex + batchSize]).to(device)
            e_i       = Variable(Noise_LDA[leftIndex:leftIndex + batchSize]).to(device)

            n_pm, n_zr = select_negative_items(noiseData.cpu().numpy(), pro_PM, pro_ZR, disease_num)
            k_i_zp = Variable(torch.tensor(n_pm + n_zr, dtype=torch.float32)).to(device)
            realData_zp = e_i * 1.0 + k_i_zp * 0.0

            # 生成器前向传播（返回 recon_h 和 total_aux_loss）
            fake_embedding, r_i, total_aux_loss  = G(Noise_LMDN, Noise_LMDN_h, noiseData, batchSize, leftIndex)

            pred_matrix = r_i * (e_i + k_i_zp)
            fake_out    = D(pred_matrix, fake_embedding)

            g_adv_loss = torch.mean(1.0 - fake_out)

            g_loss = (
                    g_adv_loss
                    + alpha * mse_loss_fn(pred_matrix, realData_zp)
                    + total_aux_loss
            )

            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        # —— 判别器训练 ——
        for _ in range(D_step):
            leftIndex = random.randint(1, lncrna_num - batchSize - 1)
            realData  = Variable(True_LDA[leftIndex:leftIndex + batchSize]).to(device)
            noiseData = Variable(Noise_LDA[leftIndex:leftIndex + batchSize]).to(device)
            e_i       = Variable(Noise_LDA[leftIndex:leftIndex + batchSize]).to(device)

            n_pm, _ = select_negative_items(noiseData.cpu().numpy(), pro_PM, pro_ZR, disease_num)
            k_i = Variable(torch.tensor(n_pm, dtype=torch.float32)).to(device)

            fake_embedding, r_i, _ = G(Noise_LMDN, Noise_LMDN_h, noiseData, batchSize, leftIndex)
            fake_out = D(r_i * (e_i + k_i), fake_embedding)

            true_embedding, _, _ = G(True_LMDN, True_LMDN_h, realData, batchSize, leftIndex)
            real_out = D(realData, true_embedding)

            real_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out))
            fake_loss = F.binary_cross_entropy(fake_out, torch.zeros_like(fake_out))
            d_loss = real_loss + fake_loss

            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        # —— 验证 ——
        labels, preds = [], []
        for testUser in range(len(Test_Adj)):
            data = Variable(Noise_LDA[testUser:testUser + 1]).to(device)
            _, predData, _ = G(Noise_LMDN, Noise_LMDN_h, data, 1, testUser)
            scores = predData[0].tolist()
            truth  = Test_Adj[testUser].tolist()
            for i, v in enumerate(truth):
                if v == 1 and [testUser, i] in test_negative:
                    labels.append(0); preds.append(scores[i])
                if v == 1 and [testUser, i] not in test_negative:
                    labels.append(1); preds.append(scores[i])

        auc  = roc_auc_score(labels, preds)
        p, r, _ = precision_recall_curve(labels, preds)
        aupr = metrics.auc(r, p)
        # 衍生评估指标（对0.5进行二值化）
        from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
        preds_binary = [1 if s >= 0.5 else 0 for s in preds]
        acc = accuracy_score(labels, preds_binary)
        mcc = matthews_corrcoef(labels, preds_binary)
        f1 = f1_score(labels, preds_binary)
        recall = recall_score(labels, preds_binary)
        precision = precision_score(labels, preds_binary)

        print(
            f"Epoch[{epoch}/{epochCount}] | AUC: {auc:.4f}, AUPR: {aupr:.4f}, ACC: {acc:.4f}, MCC: {mcc:.4f}, RECALL: {recall:.4f},F1: {f1:.4f},PRECISION: {precision:.4f}")
        # ✅ 更新各项指标最大值（单独判断）
        if auc > best_auc:
            best_auc = auc
            best_auc_labels = labels
            best_auc_preds = preds
            best_precision = precision
            torch.save(G, G_PATH)
            torch.save(D, D_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if aupr > best_aupr:
            best_aupr = aupr
        if acc > best_acc:
            best_acc = acc
        if mcc > best_mcc:
            best_mcc = mcc
        if f1 > best_f1:
            best_f1 = f1
        if recall > best_recall:
            best_recall = recall

        # ✅ 保留早停逻辑仍以 AUC 为核心
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}!")
            break
#######################################################

        scheduler_d.step(auc)
        scheduler_g.step(auc)


    # ✅ 最终返回的是每个指标的最优值（不一定同一 epoch）
    return best_auc, best_aupr, best_acc, best_mcc,best_recall, best_f1, best_precision, best_auc_labels, best_auc_preds
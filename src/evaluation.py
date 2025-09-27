## Importing libraries
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
import warnings

from .utils import my_kl_loss
from .sampling import get_prediction


## VUS Metrics Implementation
def get_anomaly_segments(labels):
    """
    Get list of anomaly segments from binary labels.
    """
    segments = []
    in_anomaly = False
    start = 0
    
    for i, label in enumerate(labels):
        if label == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif label == 0 and in_anomaly:
            segments.append((start, i))
            in_anomaly = False
    
    if in_anomaly:
        segments.append((start, len(labels)))
    
    return segments


def compute_range_auc_roc(scores, labels, buffer_size):
    """
    Compute Range-based AUC-ROC for a specific buffer size.
    """
    n = len(labels)
    extended_labels = np.zeros(n)
    
    for i in range(n):
        if labels[i] == 1:
            start = max(0, i - buffer_size // 2)
            end = min(n, i + buffer_size // 2 + 1)
            extended_labels[start:end] = 1
    
    try:
        return roc_auc_score(extended_labels, scores)
    except:
        return 0.5


def compute_range_auc_pr(scores, labels, buffer_size):
    """
    Compute Range-based AUC-PR for a specific buffer size.
    """
    n = len(labels)
    extended_labels = np.zeros(n)
    
    for i in range(n):
        if labels[i] == 1:
            start = max(0, i - buffer_size // 2)
            end = min(n, i + buffer_size // 2 + 1)
            extended_labels[start:end] = 1
    
    try:
        return average_precision_score(extended_labels, scores)
    except:
        return 0.0


def compute_vus_roc(scores, labels, max_buffer_size=None):
    """
    Compute Volume Under the Surface for ROC (VUS-ROC).
    """
    if max_buffer_size is None:
        anomaly_segments = get_anomaly_segments(labels)
        if anomaly_segments:
            max_buffer_size = int(np.median([seg[1] - seg[0] for seg in anomaly_segments]))
        else:
            max_buffer_size = 100
    
    buffer_sizes = range(1, max_buffer_size + 1)
    range_aucs = []
    
    for buffer_size in buffer_sizes:
        range_auc = compute_range_auc_roc(scores, labels, buffer_size)
        range_aucs.append(range_auc)
    
    return np.mean(range_aucs)


def compute_vus_pr(scores, labels, max_buffer_size=None):
    """
    Compute Volume Under the Surface for PR (VUS-PR).
    """
    if max_buffer_size is None:
        anomaly_segments = get_anomaly_segments(labels)
        if anomaly_segments:
            max_buffer_size = int(np.median([seg[1] - seg[0] for seg in anomaly_segments]))
        else:
            max_buffer_size = 100
    
    buffer_sizes = range(1, max_buffer_size + 1)
    range_aucs = []
    
    for buffer_size in buffer_sizes:
        range_auc = compute_range_auc_pr(scores, labels, buffer_size)
        range_aucs.append(range_auc)
    
    return np.mean(range_aucs)


def compute_add_v2(prediction, labels):
    now_anomaly_flag = False
    find_anomaly_flag = False 

    latency_list = []
    segment_len_list = []
    latency = 0
    curr_segment_len = 0

    for i, label in enumerate(labels):
        if not label:
            if now_anomaly_flag:
                latency_list.append(latency)
                segment_len_list.append(curr_segment_len)
                now_anomaly_flag = False
                find_anomaly_flag = False
                latency = 0
                curr_segment_len = 0
            else:
                pass
        else:
            now_anomaly_flag = True
            curr_segment_len += 1
            if prediction[i]:
                find_anomaly_flag = True

            if not find_anomaly_flag:
                latency += 1
            else:
                pass

    if latency > 0:
        latency_list.append(latency)
        segment_len_list.append(curr_segment_len)

    return latency_list, segment_len_list


def compute_auroc(labels, scores):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (AUROC).
    AUC-ROC is the same as AUROC.
    Args:
        labels: Ground truth labels (0 or 1).
        scores: Anomaly scores (higher = more anomalous).
    Returns:
        AUROC score.
    """
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        warnings.warn("Only one class present in labels, returning 0.5 for AUROC")
        return 0.5


def compute_auprc(labels, scores):
    """
    Compute Area Under the Precision-Recall Curve (AUPRC).
    AUC-PR is the same as AUPRC.
    Args:
        labels: Ground truth labels (0 or 1).
        scores: Anomaly scores (higher = more anomalous).
    Returns:
        AUPRC score.
    """
    try:
        return average_precision_score(labels, scores)
    except ValueError:
        warnings.warn("No positive samples in labels, returning 0.0 for AUPRC")
        return 0.0


def test(score_model,
        anomaly_ratio,
        data_loader,
        thre_loader, 
        test_loader,
        win_size,
        sde_model,
        eps, 
        svdd_criterion,
        device='cuda',
        likelihood_weighting=False):

    
    score_model.eval()
    temperature = 50

    print("======================TEST MODE======================")

    criterion = nn.MSELoss(reduce=False)

    
    attens_energy = []

    for i, (input_data, labels) in enumerate(data_loader):
        with torch.no_grad():
            input = input_data.float().to(device)
            t = torch.rand(input.shape[0], device=input.device) * (sde_model.T - eps) + eps
            z = torch.randn_like(input)
            mean, std = sde_model.marginal_prob(input, t)
            perturbed_x = mean + std[:, None, None] * z

            score, global_context, local_context, _ = score_model(perturbed_x, t)
            x_hat = get_prediction(
                            perturbed_x, 
                            t, 
                            score,
                            device=device,
                            sde_model=sde_model,
                            eps=eps)
            if not likelihood_weighting:
                loss_sde = torch.square(score * std[:, None, None, None] + z)
                loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1)
            else:
                g2 = sde_model.sde(torch.zeros_like(input), t)[1] ** 2
                loss_sde = torch.square(score + z / std[:, None, None, None])
                loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1) * g2
            
            loss_sde = loss_sde.unsqueeze(1).expand(-1, win_size)
            loss_svdd = svdd_criterion(score)
        

            loss = torch.mean(criterion(input, x_hat), dim=-1)
            gloabl_context_loss = 0.0
            local_context_loss = 0.0
            for u in range(len(local_context)):
                if u == 0:
                    gloabl_context_loss = my_kl_loss(global_context[u], (
                            local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)).detach()) * temperature
                    local_context_loss = my_kl_loss(
                        (local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        global_context[u].detach()) * temperature
                else:
                    gloabl_context_loss += my_kl_loss(global_context[u], (
                            local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)).detach()) * temperature
                    local_context_loss += my_kl_loss(
                        (local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        global_context[u].detach()) * temperature
                    
            metric = torch.softmax((-gloabl_context_loss - local_context_loss), dim=-1)
            cri = metric * loss + loss_svdd
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
    
    attens_energy = []
    
    for i, (input_data, labels) in enumerate(thre_loader):
        with torch.no_grad():
            input = input_data.float().to(device)
            t = torch.rand(input.shape[0], device=input.device) * (sde_model.T - eps) + eps
            z = torch.randn_like(input)
            mean, std = sde_model.marginal_prob(input, t)
            perturbed_x = mean + std[:, None, None] * z

            ## Score generation with score model
            score, global_context, local_context, _ = score_model(perturbed_x, t)
            x_hat = get_prediction(
                            perturbed_x, 
                            t, 
                            score,
                            device=device,
                            sde_model=sde_model,
                            eps=eps)
            if not likelihood_weighting:
                loss_sde = torch.square(score * std[:, None, None, None] + z)
                loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1)
            else:
                g2 = sde_model.sde(torch.zeros_like(input), t)[1] ** 2
                loss_sde = torch.square(score + z / std[:, None, None, None])
                loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1) * g2
            
            loss_sde = loss_sde.unsqueeze(1).expand(-1, win_size)
            loss_svdd = svdd_criterion(score)
            loss = torch.mean(criterion(input, x_hat), dim=-1)
            gloabl_context_loss = 0.0
            local_context_loss = 0.0
            for u in range(len(local_context)):
                if u == 0:
                    gloabl_context_loss = my_kl_loss(global_context[u], (
                            local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)).detach()) * temperature
                    local_context_loss = my_kl_loss(
                        (local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        global_context[u].detach()) * temperature
                else:
                    gloabl_context_loss += my_kl_loss(global_context[u], (
                            local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)).detach()) * temperature
                    local_context_loss += my_kl_loss(
                        (local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        global_context[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-gloabl_context_loss - local_context_loss), dim=-1)
            cri = metric * loss + loss_svdd
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
    # print("Threshold :", thresh)
 

    # (3) evaluation on the test set
    test_labels = []
    attens_energy = []
 
    for i, (input_data, labels) in enumerate(thre_loader):
        with torch.no_grad():
            input = input_data.float().to(device)
            t = torch.rand(input.shape[0], device=input.device) * (sde_model.T - eps) + eps
            z = torch.randn_like(input)
            mean, std = sde_model.marginal_prob(input, t)
            perturbed_x = mean + std[:, None, None] * z

            ## Score generation with score model
            score, global_context, local_context, _ = score_model(perturbed_x, t)
            x_hat = get_prediction(
                            perturbed_x, 
                            t, 
                            score,
                            device=device,
                            sde_model=sde_model,
                            eps=eps)
            if not likelihood_weighting:
                loss_sde = torch.square(score * std[:, None, None, None] + z)
                loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1)
            else:
                g2 = sde_model.sde(torch.zeros_like(input), t)[1] ** 2
                loss_sde = torch.square(score + z / std[:, None, None, None])
                loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1) * g2
            
            loss_sde = loss_sde.unsqueeze(1).expand(-1, win_size)
            loss_svdd = svdd_criterion(score)

            loss = torch.mean(criterion(input, x_hat), dim=-1)
            gloabl_context_loss = 0.0
            local_context_loss = 0.0
            for u in range(len(local_context)):
                if u == 0:
                    gloabl_context_loss = my_kl_loss(global_context[u], (
                            local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)).detach()) * temperature
                    local_context_loss = my_kl_loss(
                        (local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        global_context[u].detach()) * temperature
                else:
                    gloabl_context_loss += my_kl_loss(global_context[u], (
                            local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    win_size)).detach()) * temperature
                    local_context_loss += my_kl_loss(
                        (local_context[u] / torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        global_context[u].detach()) * temperature
            metric = torch.softmax((-gloabl_context_loss - local_context_loss), dim=-1)

            cri = metric * loss + loss_svdd
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    test_labels = np.array(test_labels)
    
    pred = (test_energy > thresh).astype(int)
    gt = test_labels.astype(int)

    # print("pred:   ", pred.shape)
    # print("gt:     ", gt.shape)

    pred_wo_adjustment = np.array(pred)
    gt_wo_adjustment = np.array(gt)
    
    # AUROC and AUPRC computation
    auroc = compute_auroc(gt, test_energy)  
    auprc = compute_auprc(gt, test_energy)
    print(f"AUC-ROC (AUROC): {auroc:.4f}, AUC-PR (AUPRC): {auprc:.4f}")
    
    # VUS metrics computation
    vus_roc = compute_vus_roc(test_energy, gt)
    vus_pr = compute_vus_pr(test_energy, gt)
    print(f"VUS-ROC: {vus_roc:.4f}, VUS-PR: {vus_pr:.4f}")
    
    ## ADD and ADP computation
    latency_list, segment_len_list = compute_add_v2(pred_wo_adjustment, gt_wo_adjustment)
    if len(latency_list) > 0:
        ADP_list = [latency_list[i] / segment_len_list[i] for i in range(len(latency_list))]
        ADP_value = sum(ADP_list) / len(ADP_list)
        ADD_value = sum(latency_list) / len(latency_list)
    else:
        ADP_value = 0.0
        ADD_value = 0.0
    print("ADD: ", ADD_value, " and ADP: ", ADP_value)

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred,
                                                                            average='binary')
    print(
        "WITH ADJUSTMENT: Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
    
    return accuracy, precision, recall, f_score, ADD_value, ADP_value, auroc, auprc, vus_roc, vus_pr, test_energy
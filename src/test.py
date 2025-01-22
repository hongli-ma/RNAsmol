import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse
#from metrics import get_cindex, get_rm2
from metrics import accuracy, sensitivity, specificity,precision, f1_score, roc_auc,pr_auc,mcc_score,recall
from dataset import *
#from model import MGraphDTA
from model import MCNN_GCN
from utils import *
from metrics import *

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    pred_cls_list = []
    label_list = []

    for data in dataloader:
        data.y = data.y.long()  
        data = data.to(device)

        with torch.no_grad():
            ligand_x,protein_x,f,pred = model(data)
            loss = criterion(pred, data.y)
            pred_cls = torch.argmax(pred, dim=-1)

            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(data.y.detach().cpu().numpy())
            running_loss.update(loss.item(), data.y.size(0))

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc = accuracy(label, pred_cls)
    sen = sensitivity(label,pred_cls)
    spe = specificity(label,pred_cls)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    f1score=f1_score(label,pred_cls)
    rocauc = roc_auc(label, pred)
    prauc=pr_auc(label,pred)
    mcc=mcc_score(label,pred_cls)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred, f


def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='human or celegans')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    parser.add_argument('--model',type=int,default=0,help='training model')
    args = parser.parse_args()

    data_root = "data"
    DATASET = args.dataset
    model_path = args.model_path
    training_model=args.model

    fpath = os.path.join(data_root, DATASET)

    test_set = GNNDataset(fpath, types='test')
    print("Number of test: ", len(test_set))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MCNN_GCN(3, 25 + 1, embedding_size=96, filter_num=32, out_dim=2,ban_heads=2).to(device)

    criterion = nn.CrossEntropyLoss()
    load_model_dict(model, model_path)

    test_loss, test_acc, test_sen, test_spe, test_pre, test_rec, test_f1, test_rocauc, test_prauc, test_mcc, test_label, test_pred, test_att = val(model, criterion, test_loader, device)
    msg = "test_loss-%.4f, test_acc-%.4f, test_sen-%.4f, test_spe-%.4f, test_pre-%.4f, test_rec-%.4f, test_f1-%.4f, test_roauc-%.4f, test_prauc-%.4f, test_mcc-%.4f" % (test_loss, test_acc, test_sen, test_spe, test_pre, test_rec, test_f1, test_rocauc, test_prauc, test_mcc)

    print(msg)

if __name__ == "__main__":
    main()

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
from metrics import accuracy, sensitivity, specificity, precision, f1_score, roc_auc, pr_auc,mcc_score, recall
from dataset import *
from model import MCNN_GCN
from utils import *
from log.train_logger import TrainLogger


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
    prauc=pr_auc(label, pred)
    mcc=mcc_score(label,pred_cls)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred,f



def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='human or celegans')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
    parser.add_argument('--model',type=int,default=0,help='training model')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        training_model=args.model
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)
    training_model=params.get("training_model")

    train_set = GNNDataset(fpath, types='train')
    val_set = GNNDataset(fpath, types='val')
    test_set = GNNDataset(fpath, types='test')

    logger.info(f"Number of train: {len(train_set)}")
    logger.info(f"Number of val: {len(val_set)}")
    logger.info(f"Number of test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=8)
    

    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    model = MCNN_GCN(3, 26, embedding_size=96, filter_num=32, out_dim=2,ban_heads=2).to(device)

    epochs = 120

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()


    running_loss = AverageMeter()

    model.train()

    best_val_auc = -1
    best_val_loss = 100

    for epoch in range(epochs):

        for batch_idx, data in enumerate(train_loader):
          
            data.y = data.y.long()    
            data = data.to(device)
            ligand_x,protein_x,f,pred = model(data)

            loss = criterion(pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 


        epoch_loss = running_loss.get_average()
        running_loss.reset()

        val_loss, val_acc, val_sen, val_spe, val_pre, val_rec, val_f1, val_rocauc, val_prauc, val_mcc, val_label, val_pred, val_att = val(model, criterion, val_loader, device)
        test_loss, test_acc, test_sen, test_spe, test_pre, test_rec, test_f1, test_rocauc, test_prauc, test_mcc, test_label, test_pred, test_att = val(model, criterion, test_loader, device)

        msg = "epoch-%d, loss-%.4f, val_loss-%.4f, val_acc-%.4f, val_f1-%.4f, val_rocauc-%.4f, val_prauc-%.4f, val_mcc-%.4f, test_loss-%.4f, test_acc-%.4f, test_sen-%.4f, test_spe-%.4f,test_pre-%.4f, test_rec-%.4f, test_f1-%.4f, test_rocauc-%.4f, test_prauc-%.4f, test_mcc-%.4f" % (epoch, epoch_loss, val_loss, val_acc,val_f1,val_rocauc, val_prauc, val_mcc, test_loss, test_acc, test_sen, test_spe, test_pre, test_rec, test_f1, test_rocauc,test_prauc,test_mcc) 
        logger.info(msg)

        if val_loss < best_val_loss and val_rocauc > best_val_auc:
             best_val_auc=val_rocauc
             best_val_loss=val_loss
             
             np.save(logger.get_model_dir()+"/best_model_epoch"+str(epoch)+"_val_label.npy",val_label)
             np.save(logger.get_model_dir()+"/best_model_epoch"+str(epoch)+"_val_pred.npy",val_pred)

             np.save(logger.get_model_dir()+"/best_model_epoch"+str(epoch)+"_test_label.npy",test_label)
             np.save(logger.get_model_dir()+"/best_model_epoch"+str(epoch)+"_test_pred.npy",test_pred)
            
             torch.save(val_att,logger.get_model_dir()+"/best_model_epoch"+str(epoch)+"_val_att.pt")
             torch.save(test_att,logger.get_model_dir()+"/best_model_epoch"+str(epoch)+"_test_att.pt")
              
             if save_model:
                save_model_dict(model, logger.get_model_dir(), "epoch-%d" % (epoch))


if __name__ == "__main__":
    main()

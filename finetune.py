import torch
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_scaffold_split, random_split
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
from util import parse_config

from loguru import logger

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(config, epoch, model, device, loader, optimizer):
    model.train()
    epoch_iter = tqdm(loader, desc="train iter")
    for it, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        pred, node_representation = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y + 1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))  
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")


def eval(config, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for it, batch in enumerate(tqdm(loader, desc="valid iter")):
        batch = batch.to(device)

        with torch.no_grad():
            pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)


def main(config):

    torch.manual_seed(config.runseed)
    np.random.seed(config.runseed)
    device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.runseed)

    if config.dataset == "sider":
        num_tasks = 27
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("./dataset/" + config.dataset, dataset=config.dataset)
    print(dataset)

    smiles_list = pd.read_csv('./dataset/' + config.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    print("scaffold")

    print('++++++++++', train_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers = config.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers = config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers = config.num_workers)

    #set up model
    model = GNN_graphpred(config.num_layer, config.emb_dim, num_tasks, JK = config.JK, drop_ratio = config.dropout_ratio, graph_pooling = config.graph_pooling, gnn_type = config.gnn_type)
    if not config.input_model_file == "None":
        print('Not from scratch')
        model.from_pretrained('model_gin/{}.pth'.format(config.input_model_file))
    
    model.to(device)
    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if config.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":config.lr*config.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":config.lr*config.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=config.lr, weight_decay=config.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    if not config.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(config.runseed) + '/' + config.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, config.epochs+1):
        print("====epoch " + str(epoch))
        
        train(config, epoch, model, device, train_loader, optimizer)

        print("====Evaluation")
        if config.eval_train:
            train_acc = eval(config, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(config, model, device, val_loader)
        test_acc = eval(config, model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not config.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

    print('Best epoch:', val_acc_list.index(max(val_acc_list)))
    print('Best auc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])

    exp_path = os.getcwd() + '/{}_results/{}/'.format(config.input_model_file, config.dataset)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    df = pd.DataFrame({'train':train_acc_list,'valid':val_acc_list,'test':test_acc_list})
    df.to_csv(exp_path + 'seed{}.csv'.format(config.runseed))

    logs = 'Dataset:{}, Seed:{}, Best Epoch:{}, Best Acc:{:.5f}'.format(config.dataset, config.runseed, val_acc_list.index(max(val_acc_list)), test_acc_list[val_acc_list.index(max(val_acc_list))])
    with open(exp_path + '{}_log.csv'.format(config.dataset),'a+') as f:
        f.write('\n')
        f.write(logs)

    if not config.filename == "":
        writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dataset/sider/config.yaml')

    args = parser.parse_args()
    config = parse_config(args.config)
    main(config)

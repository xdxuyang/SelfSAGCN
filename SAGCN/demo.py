from __future__ import division
from __future__ import print_function
import math
import time
import argparse
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data_all, accuracy
from models import GCN

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

test1 = np.zeros(10)

for t in range(10):

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

            # Load data

            # adj, features, labels, idx_train, idx_val, idx_test,adj_cnn= load_data_stl10()
    adj, features, labels, idx_train, idx_val, idx_test, adj_cnn = load_data_all('cora')

    if len(labels.size()) == 1:
        target = torch.zeros(labels[idx_train].size(0), 7).scatter_(1, labels[idx_train].view(-1, 1), 1)



    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)

    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            # c= labels[idx_test]
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        target =target.cuda()
        adj_cnn = adj_cnn.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    losses_vae = np.empty((2000,))

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output, feature, cnn_output, cnn_feature, gcn_features1, cnn_features1 = model(features, adj, adj_cnn)
        lam = adaptation_factor(epoch / 50)
        loss_train = 1 * model.loss(output[idx_train], target) + 1 * model.loss(cnn_output[idx_train],target)
        semantic_loss_1 =  model.adloss1(cnn_features1[idx_train], gcn_features1, labels[idx_train], output)

        semantic_loss_2 =  model.adloss(cnn_feature[idx_train], feature, labels[idx_train], output)

        semantic_loss = 10*semantic_loss_1+2.5*semantic_loss_2



        loss_train = 1* loss_train +lam*semantic_loss
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        if not args.fastmode:

            model.eval()
            output, feature, cnn_output, cnn_feature, gcn_features1, cnn_features1 = model(features, adj,adj_cnn)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print('Epoch: {:04d}'.format(epoch),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_test: {:.4f}'.format(acc_test.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t),
                'semantic_loss: {:.4f}'.format(semantic_loss_1.item()))
        losses_vae[epoch] = acc_val.item()

        return acc_val


    def test():
        model.eval()
        output, feature, cnn_output, cnn_feature, gcn_features1, cnn_features1 = model(features, adj, adj_cnn)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))
        print(idx_train)

        return acc_test


    def adaptation_factor(x):
        den = 1.0 + math.exp(-10 * x)
        lamb = 2.0 / den - 1.0
        return min(lamb, 1.0)


    best_acc = 0
    best_epoch = 0
    bad_counter = 0

    t_total = time.time()

    for epoch in range(args.epochs):
        acc_val = train(epoch)

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if acc_val >= best_acc:
            best_acc = acc_val
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 500:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)


    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))


    test1[t] = test()
print('acc_train: {:.4f}'.format(np.mean(test1)))






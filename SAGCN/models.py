import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout
        self.cudable = True
        self.decay = 0.3
        self.n_class = nclass
        self.s_centroid = torch.zeros(nclass, nclass)
        self.s_centroid = self.s_centroid.cuda()
        self.t_centroid = torch.zeros(nclass, nclass)
        self.t_centroid = self.t_centroid.cuda()

        self.s1_centroid = torch.zeros(nclass, nhid)
        self.s1_centroid = self.s1_centroid.cuda()
        self.t1_centroid = torch.zeros(nclass, nhid)
        self.t1_centroid = self.t1_centroid.cuda()



        self.MSEloss =  nn.MSELoss()





    def forward(self, x, adj,adj_CNN):
        x_gcn = F.relu(self.gc1(x, adj))
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        gcn_features1 = x_gcn
        x_gcn = self.gc2(x_gcn, adj)
        gcn_features = x_gcn




        x_cnn = F.relu(self.gc1(x, adj_CNN))
        x_cnn = F.dropout(x_cnn, self.dropout, training=self.training)
        cnn_features1 = x_cnn
        x_cnn = self.gc2(x_cnn, adj_CNN)
        cnn_features = x_cnn
        return F.log_softmax(x_gcn, dim=1),gcn_features,F.log_softmax(x_cnn, dim=1),cnn_features,gcn_features1,cnn_features1


    def loss(self,input,label):


        return -torch.mean(torch.sum(input*label, dim=1))

    def adloss(self, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        s_labels = y_s

        tlabel = F.softmax(y_t, dim=1)+0 * torch.randn(y_t.size()).cuda()

        # get labels
        t_labels = torch.max(tlabel, 1)[1]


        # image number in each class
        ones_s = torch.ones_like(s_labels, dtype=torch.float)
        ones_t = torch.ones_like(t_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones_s)
        t_n_classes = zeros.scatter_add(0, t_labels, ones_t)

        # image number cannot be 0, when calculating centroids
        ones_s = torch.ones_like(s_n_classes)
        ones_t = torch.ones_like(t_n_classes)
        s_n_classes = torch.max(s_n_classes, ones_s)
        t_n_classes = torch.max(t_n_classes, ones_t)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, d)
        if self.cudable:
            zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))

        # Moving Centroid
        decay = self.decay
        s_centroid = (1 - decay) * self.s_centroid + decay * current_s_centroid
        t_centroid = (1 - decay) * self.t_centroid + decay * current_t_centroid
        semantic_loss = self.MSEloss(s_centroid, t_centroid)
        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()




        return semantic_loss


    def adloss1(self, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        # get labels
        s_labels = y_s

        tlabel = F.softmax(y_t, dim=1)+0 * torch.randn(y_t.size()).cuda()

        # get labels
        t_labels = torch.max(tlabel, 1)[1]


        # image number in each class
        ones_s = torch.ones_like(s_labels, dtype=torch.float)
        ones_t = torch.ones_like(t_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones_s)
        t_n_classes = zeros.scatter_add(0, t_labels, ones_t)

        # image number cannot be 0, when calculating centroids
        ones_s = torch.ones_like(s_n_classes)
        ones_t = torch.ones_like(t_n_classes)
        s_n_classes = torch.max(s_n_classes, ones_s)
        t_n_classes = torch.max(t_n_classes, ones_t)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, d)
        if self.cudable:
            zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        current_s1_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1))
        current_t1_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))

        # Moving Centroid
        decay = self.decay
        s1_centroid = (1 - decay) * self.s1_centroid + decay * current_s1_centroid
        t1_centroid = (1 - decay) * self.t1_centroid + decay * current_t1_centroid
        semantic_loss = self.MSEloss(s1_centroid, t1_centroid)
        self.s1_centroid = s1_centroid.detach()
        self.t1_centroid = t1_centroid.detach()




        return semantic_loss



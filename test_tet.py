from dgl.data import TUDataset
from dgl.nn.pytorch.glob import AvgPooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from dgl.dataloading import GraphDataLoader
from torch.utils.data import random_split


class GCN(torch.nn.Module):
    """3层GCN+1层线性层"""

    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid, activation=F.relu)
        self.gc2 = GraphConv(nhid, nhid, activation=F.relu)
        self.gc3 = GraphConv(nhid, nhid, activation=F.relu)
        self.lin = nn.Linear(nhid, nclass)
        self.avgpool = AvgPooling()
        self.dropout = dropout

    def forward(self, g, features):
        """模型前向传播"""
        h = self.gc1(g, features)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gc2(g, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gc3(g, h)
        h = self.avgpool(g, h)
        h = self.lin(h)
        return F.log_softmax(h, dim=1)


def train(model, lr=0.001, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        loss_all = 0
        for batch_graphs, batch_labels in train_loader:
            # 每个批次里获取特征和处理标签
            batch_graphs = batch_graphs.to(device)
            features = batch_graphs.ndata['node_attr'].float()
            batch_labels = batch_labels.squeeze(dim=1)
            batch_labels = torch.Tensor(batch_labels).long()

            optimizer.zero_grad()
            # 前向传播
            preds = model(batch_graphs, features)
            # 计算损失
            loss = F.nll_loss(preds, batch_labels)
            loss.backward()
            loss_all += loss.item() * batch_labels.shape[0]
            # 反向传播
            optimizer.step()

        loss_train = loss_all / train_num
        if epoch % 100 == 0:
            print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss_train))


def test(model, valid_loader):
    model.eval()
    correct = 0
    for batch_graphs, batch_labels in valid_loader:
        batch_graphs = batch_graphs.to(device)
        features = batch_graphs.ndata['node_attr'].float()
        batch_labels = batch_labels.squeeze(dim=1)

        pred = model(batch_graphs, features)
        correct += float(torch.sum(torch.argmax(pred, dim=1) == batch_labels).item())
    return correct / test_num


if __name__ == '__main__':
    dataset = TUDataset(name='ENZYMES', raw_dir='./data/ENZYMES')
    # g, labels = dataset

    # 获取train和test的样本数量
    train_num = int(0.8 * len(dataset))
    test_num = int(0.2 * len(dataset))

    # 使用Pytorch的random_split获取train和test的Dataset类
    train_dataset, test_dataset = random_split(dataset, [train_num, test_num])

    # 使用DGL的GraphDataLoader构建迷你批次数据
    train_loader = GraphDataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = GraphDataLoader(test_dataset, batch_size=64)
    # 从一个样本图里获取特征维度
    g0, lable0 = dataset[0]
    nfeat = g0.ndata['node_attr'].shape[1]
    nclass = lable0.max().numpy() + 1
    nhid = 64
    device = 'cpu'
    # device = ‘cuda’ # 如果有CUDA的话

    model = GCN(nfeat, nhid, nclass).to(device)
    train(model, epochs=1000)

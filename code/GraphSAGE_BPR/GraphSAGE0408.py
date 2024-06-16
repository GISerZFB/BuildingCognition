import os
import time
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GraphSAGE, GAT, GCN
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from get_graph import get_Graph_data
import matplotlib.pyplot as plt
import geopandas as gpd

# 配置项
class configs():
    def __init__(self):
        # Data
        self.save_model_dir = './'
        self.seed = 2024

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        torch.cuda.set_device(0)

        torch.cuda.is_available()
        torch.cuda.current_device()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset, self.Geosdata, self.train_val_test = get_Graph_data(0.2, 0.2)
        self.dataset.to(self.device)

        self.epoch = 500
        self.in_features = 5  # 输入层 特征大小
        self.hidden_features = 8  # 隐层大小
        self.hidden_nums = 2  # 隐层数量
        self.output_features = 2  # 输出层大小

        self.learning_rate = 0.01
        self.weight_decay = 0
        self.dropout = 0.3

        self.istrain = True
        self.istest = True
        self.isembedding = True

cfg = configs()

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(seed=cfg.seed)

class myGraphSAGE_run():
    def train(self):
        t = time.time()
        dataset = cfg.dataset
        model = GraphSAGE(in_channels=cfg.in_features,
                          hidden_channels=cfg.hidden_features,
                          num_layers=cfg.hidden_nums,
                          out_channels=cfg.output_features,
                          dropout=cfg.dropout,
                          act='relu',
                          aggr='max').to(cfg.device)
        # model = GraphSAGE1(dataset.num_features, cfg.output_features).to(cfg.device)
        print(model)
        data = dataset
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        model.train()
        self.Train_Loss = []
        self.Train_Acc = []
        self.Val_Loss = []
        self.Val_Acc = []
        for epoch in range(cfg.epoch):
            # 训练
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            # print(output.long())
            preds = output.max(dim=1)[1]

            loss_train = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
            correct = preds[data.train_mask].eq(data.y[data.train_mask]).sum().item()
            acc_train = correct / int(data.train_mask.sum())

            loss_train.backward()
            optimizer.step()

            # 验证
            loss_val = F.cross_entropy(output[data.val_mask], data.y[data.val_mask])
            correct = preds[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            acc_val = correct / int(data.val_mask.sum())

            self.Train_Loss.append(loss_train.item())
            self.Train_Acc.append(acc_train)
            self.Val_Loss.append(loss_val.item())
            self.Val_Acc.append(acc_val)
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'time: {:.4f}s'.format(time.time() - t))
        torch.save(model, os.path.join(cfg.save_model_dir, f'GraphSAGE_{cfg.hidden_nums}.pth'))  # 模型保存

    def infer(self):
        # Create Test Processing
        dataset = cfg.dataset
        data = dataset
        model_path = os.path.join(cfg.save_model_dir, f'GraphSAGE_{cfg.hidden_nums}.pth')
        model = torch.load(model_path, map_location=torch.device(cfg.device))
        model.eval()
        output = model(data.x, data.edge_index)
        params = sum(p.numel() for p in model.parameters())
        preds = output.max(dim=1)[1]
        loss_test = F.cross_entropy(output[data.test_mask], data.y[data.test_mask])
        correct = preds[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc_test = correct / int(data.test_mask.sum())
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test),
              'params={:.4f}k'.format(params / 1024))

        return preds.cpu().numpy()

    def Com(self):
        epochs = np.arange(1, cfg.epoch + 1)
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("value")
        plt.plot(epochs, self.Train_Loss, color="b", label="Train Loss")
        plt.plot(epochs, self.Val_Loss, color="g", label="Val Loss")
        plt.legend()
        plt.show()

        plt.title("Acc Curve")
        plt.xlabel("Epoch")
        plt.ylabel("value")
        plt.plot(epochs, self.Train_Acc, color="b", label="Train Acc")
        plt.plot(epochs, self.Val_Acc, color="g", label="Val Acc")
        plt.legend()
        plt.show()

    def get_embedding(self):
        # Create Test Processing
        dataset = cfg.dataset
        data = dataset
        model_path = os.path.join(cfg.save_model_dir, f'GraphSAGE_{cfg.hidden_nums}.pth')
        model = torch.load(model_path, map_location=torch.device(cfg.device))
        model.eval()
        with torch.no_grad():  # 关闭梯度计算
            _, GraphSAGE_embeddings = model(data)
        print("GraphSAGE_embeddings :",
              GraphSAGE_embeddings.detach().cpu().numpy().shape)  # print : "node2vec_embeddings : (2708, 64)"

if __name__ == '__main__':
    mygraph = myGraphSAGE_run()
    if cfg.istrain == True:
        mygraph.train()

        np.savetxt(f"GraphSAGE_TrainLoss_{cfg.hidden_nums}.txt", np.array(mygraph.Train_Loss), fmt='%.8f')
        np.savetxt(f"GraphSAGE_ValLoss_{cfg.hidden_nums}.txt", np.array(mygraph.Val_Loss), fmt='%.8f')
    if cfg.istest == True:
        pre = mygraph.infer()
        np.savetxt(f"GraphSAGE_ValAcc_{cfg.hidden_nums}.txt", np.array(mygraph.Val_Acc), fmt='%.8f')

        data = {"prelabel": pd.Series(pre, dtype=np.int32),
                "flag": pd.Series(cfg.train_val_test, dtype=np.int32)}
        df = gpd.GeoDataFrame(data, geometry=cfg.Geosdata)
        df.to_file(f'../../data/newshanghai/GraphSAGE_pre_{cfg.hidden_nums}.shp')

    # mygraph.show()

    # if cfg.isembedding == True:
    #     mygraph.get_embedding()
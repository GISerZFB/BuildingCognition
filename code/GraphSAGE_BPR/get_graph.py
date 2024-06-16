import geopandas as gpd
import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import math
'''
新解决的问题：
1、任意输入障碍要素构网错误的问题！
2、修改直角化处理方式为 以开口边的特征点数量为约束进行  尚未开始！！！！！
'''
def cal_d(pt1, pt2):
    return math.sqrt((pt2[1] - pt1[1]) ** 2 + (pt2[0] - pt1[0]) ** 2)

def index_to_mask(index, size):
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    return mask

def get_Graph_data(train_ratio=0.2, val_ratio=0.2):
    DATA = pd.read_csv(f'data/R.csv')
    try_dataframe = gpd.read_file(f'data/experimentdata.shp')
    Com = try_dataframe['Com'].values
    Ecc = try_dataframe['Ecc'].values
    Rec = try_dataframe['Rec'].values
    Con = try_dataframe['Con'].values
    Num = try_dataframe['Num'].values
    SimGSC = try_dataframe['SimGSC'].values

    Dis = try_dataframe['Dis'].values
    Area = try_dataframe['Area'].values
    Direct = try_dataframe['Direct'].values
    SimK1 = try_dataframe['SimK1'].values
    FPro = try_dataframe['FPro'].values
    Label = try_dataframe['label'].values
    Label = np.nan_to_num(Label).astype(np.int16)
    Geos = try_dataframe['geometry']

    SimK1 = (SimK1 - np.min(SimK1)) / (np.max(SimK1) - np.min(SimK1))

    '''构建图结构 以及用于可视化操作'''
    G = nx.Graph()
    for index, line in enumerate(Geos):
        G.add_node(index, label=Label[index])

    for index, line in enumerate(Geos):
        NEAR_FID = DATA.loc[(DATA.IN_FID == index)]['NEAR_FID'].tolist()
        for ii, j in enumerate(NEAR_FID):
            if G.has_edge(index, j):
                continue
            G.add_edge(index, j)
    print(f'图节点数量：{len(G.nodes)}')
    print(f'图边数量：{len(G.edges)}')

    '''构建训练 图数据结构'''
    edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)  # Adjacency Matrix

    features_M = []
    label = []
    for node in G.nodes:
        temp = [Dis[node],
                Area[node],
                Direct[node],
                FPro[node],
                SimK1[node]]
        label.append(Label[node])
        features_M.append(temp)
    x = torch.tensor(features_M, dtype=torch.float32)
    y = torch.tensor(label, dtype=torch.long)

    '''设置训练数据和验证数据的比例'''
    # train_ratio = 0.5
    node_num = len(G.nodes)  # 计算训练样本数
    val_num = int(node_num * val_ratio)
    train_num = int(node_num * train_ratio)
    test_num = node_num - val_num - train_num

    idx_test = range(0, test_num)
    print(f'测试集数量：{test_num}')
    idx_val = range(test_num, test_num+val_num)
    print(f'验证集数量：{val_num}')
    idx_train = range(test_num+val_num, test_num+val_num+train_num)
    print(f'训练集数量：{train_num}')

    train_mask = torch.tensor(index_to_mask(idx_train, size=node_num), dtype=torch.bool)
    val_mask = torch.tensor(index_to_mask(idx_val, size=node_num), dtype=torch.bool)
    test_mask = torch.tensor(index_to_mask(idx_test, size=node_num), dtype=torch.bool)

    Graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # 标记训练集、验证集和测试集
    train_val_test = np.zeros(node_num, dtype=int)
    train_val_test[idx_train] = 1
    train_val_test[idx_val] = 2
    train_val_test[idx_test] = 3
    return Graph_data, try_dataframe['geometry'], train_val_test

if __name__ == '__main__':
    data = get_Graph_data(0.3, 0.1)







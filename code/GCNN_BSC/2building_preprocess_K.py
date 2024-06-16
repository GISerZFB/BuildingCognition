import time
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import math
from Douglas_Peucker import DouglasPeuker
from shapely.geometry import *
import networkx as nx
from triangle import triangulate
import matplotlib
from matplotlib.patches import Ellipse
from torch_geometric.data import Data
import torch
from torch_geometric.data import InMemoryDataset, Dataset
KK = 2

# 计算两个点之间的距离 非Point
def cal_d(pt1, pt2):
    return math.sqrt((pt2[1] - pt1[1]) ** 2 + (pt2[0] - pt1[0]) ** 2)
def cal_d_xy(pt1, pt2):
    return math.sqrt((pt2.y - pt1.y) ** 2 + (pt2.x - pt1.x) ** 2)

# 计算起始点方向
def cal_angle(pt1, pt2):
    arc = math.atan2(pt2[1]-pt1[1], pt2[0]-pt1[0])
    if arc < 0:
        arc = arc + 2*math.pi
    return arc / math.pi * 180

# 加密线段用
def encription(pt1, pt2, ls):
    pts =[pt1]
    d = cal_d(pt1, pt2)
    mean_len = d / (ls + 1)
    a = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    for j in range(ls):
        x = pt1[0] + mean_len * (j + 1) * math.cos(a)
        y = pt1[1] + mean_len * (j + 1) * math.sin(a)
        pt = (x, y)
        pts.append(pt)

    return pts

def cal_L_angle(pt11, pt12, pt21, pt22):
    a = cal_d(pt11, pt12)
    b = cal_d(pt21, pt22)
    cosab = ((pt11[0] - pt12[0]) * (pt21[0] - pt22[0]) + (pt11[1] - pt12[1]) * (pt21[1] - pt22[1])) / (a * b)
    sinab = 0
    if cosab == 1:
        sinab = 0
    else:
        sinab = math.sqrt(abs(1 - cosab ** 2))
    # if cosab < 0:
    #     sin2 = -sin2
    return sinab, cosab

def cal_vector_angle(pt, pt1, pt2):
    a = cal_d(pt, pt1)
    b = cal_d(pt, pt2)
    return ((pt1[0] - pt[0]) * (pt2[0] - pt[0]) + (pt1[1] - pt[1]) * (pt2[1] - pt[1])) / (a * b)

def cal_sincosA(pt, pt1, pt2, geo, tri):
    cosa = cal_vector_angle(pt, pt1, pt2)
    sina = 0
    if cosa < -0.999 or cosa > 0.999:
        pass
    else:
        angle = math.acos(cosa)
        if geo.contains(tri.centroid):
            pass
        else:
            angle = 2 * math.pi - angle
        sina = math.sin(angle)
    return sina, cosa

# 计算一阶邻域特征
def cal_K1(FID, geo, pt, shape):
    pt1 = []
    pt2 = []
    if FID == 0:
        pt1 = shape[-1]
        pt2 = shape[FID + 1]
    elif FID == len(shape) - 1:
        pt1 = shape[FID - 1]
        pt2 = shape[0]
    else:
        pt1 = shape[FID - 1]
        pt2 = shape[FID + 1]
    tri = Polygon([pt1, pt, pt2])
    sinA, cosA = cal_sincosA(pt, pt1, pt2, geo, tri)
    pt = Point(pt)
    pt0 = geo.centroid
    ptt = tri.centroid

    cosB = cal_vector_angle([pt.x, pt.y], [pt0.x, pt0.y], [ptt.x, ptt.y])
    sinB = math.sqrt(abs(1 - cosB ** 2))

    d1 = cal_d_xy(ptt, pt)
    d2 = cal_d_xy(pt0, pt)
    d3 = cal_d_xy(pt0, ptt)
    d4 = cal_d(pt1, pt2)
    if geo.contains(tri.centroid) is False:
        d1 = -d1
    return [sinA, cosA, sinB, cosB, d1, d2, d3, d4]

# 计算二阶邻域特征
def cal_K2(FID, geo, pt, shape):
    pt1 = []
    pt2 = []
    if FID == 0:
        pt1 = shape[-2]
        pt2 = shape[2]
    elif FID == len(shape) - 1:
        pt1 = shape[-3]
        pt2 = shape[1]
    elif FID == len(shape) - 2:
        pt1 = shape[FID-2]
        pt2 = shape[0]
    elif FID == 1:
        pt1 = shape[-1]
        pt2 = shape[3]
    else:
        pt1 = shape[FID - 2]
        pt2 = shape[FID + 2]
    tri = Polygon([pt1, pt, pt2])
    sinA, cosA = cal_sincosA(pt, pt1, pt2, geo, tri)
    pt = Point(pt)
    pt0 = geo.centroid
    ptt = tri.centroid

    cosB = cal_vector_angle([pt.x, pt.y], [pt0.x, pt0.y], [ptt.x, ptt.y])
    sinB = math.sqrt(abs(1 - cosB ** 2))

    d1 = cal_d_xy(ptt, pt)
    d2 = cal_d_xy(pt0, pt)
    d3 = cal_d_xy(pt0, ptt)
    d4 = cal_d(pt1, pt2)
    if geo.contains(tri.centroid) is False:
        d1 = -d1
    return [sinA, cosA, sinB, cosB, d1, d2, d3, d4]

def get_data():
    dataframe = gpd.read_file('../../data/newshanghai/上海市局部.shp', encode='utf-8')
    # dataframe = gpd.read_file('../data/nanjing_test/nanjing_test_P.shp', encode='utf-8')
    geo_shape = dataframe['geometry']

    data_list = []
    for index, cell in enumerate(geo_shape):
        print(f'---{index}---')

        # ############### 建筑物多边形插值处理 ###############
        shape = list(cell.exterior.coords)[0:-1]

        d = DouglasPeuker()
        shape = d.main(shape, 0.05)
        geo = Polygon(shape)

        # ############### 建筑物 图结构 构建 ##################
        G = nx.Graph()
        for FID, pt in enumerate(shape):

            K1 = cal_K1(FID, geo, pt, shape)
            K2 = cal_K2(FID, geo, pt, shape)
            G.add_node(FID, pt=pt, K1=K1, K2=K2)

        temp = []
        for FID in range(len(shape) - 1):
            temp.append([FID, FID+1])
        temp.append([0, len(shape) - 1])

        # 利用 NetWorkX 建立图的边索引
        G.add_edges_from(temp)
        # gpd.GeoSeries(geo).plot()
        # plt.show()
        # nx.draw(G, with_labels=True)
        # plt.show()

        # 图 边
        edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)  # Adjacency Matrix

        # 计算边权重
        edge_attr = []
        for cell in G.edges:
            dd = cal_d(shape[cell[0]], shape[cell[1]])
            G[cell[0]][cell[1]]['weight'] = dd
            edge_attr.append(dd)
        edge_attr = np.array(edge_attr)
        edge_attr = edge_attr / np.max(edge_attr)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # nx.draw(G, with_labels=True)
        # plt.show()

        features_F = []
        for node in G.nodes:
            if KK == 1:
                temp = G.nodes[node]['K1']
            else:
                temp = G.nodes[node]['K1'] + G.nodes[node]['K2']
            features_F.append(temp)
        features_F = np.array(features_F, np.float64)

        # 计算最小外接矩形长轴长度
        boundary_xy = list(geo.minimum_rotated_rectangle.boundary.coords)
        L_value = max([cal_d(boundary_xy[0], boundary_xy[1]), cal_d(boundary_xy[1], boundary_xy[2])])

        features_F[:, 4] = features_F[:, 4] / L_value
        features_F[:, 5] = features_F[:, 5] / L_value
        features_F[:, 6] = features_F[:, 6] / L_value
        features_F[:, 7] = features_F[:, 7] / L_value
        if KK == 2:
            features_F[:, 12] = features_F[:, 12] / L_value
            features_F[:, 13] = features_F[:, 13] / L_value
            features_F[:, 14] = features_F[:, 14] / L_value
            features_F[:, 15] = features_F[:, 15] / L_value

        x = torch.tensor(np.array(features_F), dtype=torch.float)

        C1 = (4 * math.pi * geo.area) / geo.length ** 2  # 圆形度
        C2 = geo.area / geo.convex_hull.area  # 紧凑度
        E = min([cal_d(boundary_xy[0], boundary_xy[1]), cal_d(boundary_xy[1], boundary_xy[2])]) / L_value  # 偏心率
        R = geo.area / geo.minimum_rotated_rectangle.area  # 矩形度

        C2ER = torch.tensor(np.array([C1, C2, E, R]), dtype=torch.float)
        # print(C1, C2, E, R)
        # gpd.GeoSeries(geo).plot()
        # plt.show()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, C2ER=C2ER)  # 创建Features
        data_list.append(data)

    return data_list

data_list_ = get_data()

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.
        data_list = data_list_

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    a = MyOwnDataset(f"./Shanghai_K{KK}")




import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def is_outer_rectangle_near(m, n, d):
    # (minx, miny, maxx, maxy)
    if m[1] > n[3] + d or m[0] > n[2] + d or m[2] + d < n[0] or m[3] + d < n[1]:
        return False
    else:
        return True

def cal_matrix(TP, TN, FP, FN):
    Accuracy = (TP + TN) / (TP + FN + FP + TN)
    Precion = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precion * Recall) / (Precion + Recall)
    print(f"Accuracy: {Accuracy}\nPrecion:{Precion}\nRecall:{Recall}\nF1:{F1}")
    return Accuracy, Precion, Recall, F1

def cal_my_method():
    y_true = []
    y_pred = []
    Graph_dataframe = gpd.read_file(f'data/GraphSAGE_pre_2.shp')
    data_dataframe = gpd.read_file(f'data/experimentdata.shp')

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    stardard_label = data_dataframe['label'].values

    flag = Graph_dataframe['flag'].values
    for index, cell in enumerate(Graph_dataframe['prelabel'].values):
        if flag[index] == 1:
            continue
        label = stardard_label[index]

        y_pred.append(cell)
        y_true.append(label)

        if cell == 1 and label == 1:
            TP += 1
        elif cell == 0 and label == 1:
            FN += 1
        elif cell == 0 and label == 0:
            TN += 1
        elif cell == 1 and label == 0:
            FP += 1
        else:
            print('error!')
    print('My Method:')
    cal_matrix(TP, TN, FP, FN)
    return y_true, y_pred


if __name__ == '__main__':
    y_true1, y_pred1 = cal_my_method()
    print('================')
    print('my method')
    print('acc', accuracy_score(y_true1, y_pred1))
    print('pre', precision_score(y_true1, y_pred1))
    print('rec', recall_score(y_true1, y_pred1))
    print('fa', f1_score(y_true1, y_pred1))
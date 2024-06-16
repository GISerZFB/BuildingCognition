from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling, SAGPooling, SAGEConv, GCNConv
import torch.nn.functional as F
import torch
import numpy as np
import geopandas as gpd

from torch_geometric.data import InMemoryDataset
from shape_encode.Visualiation import TSE_visual

class Net(torch.nn.Module):
    def __init__(self, K=1):

        super(Net, self).__init__()
        self.conv1 = GCNConv(8*K, 32)
        self.conv2 = GCNConv(32, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 32)

        self.lin1 = torch.nn.Linear(68, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index, edge_weight, batch, C2ER = data.x, data.edge_index, data.edge_attr, data.batch, data.C2ER
        x = F.relu(self.conv1(x, edge_index, edge_weight))  # n*128
        x = self.dropout(x)
        x0 = gmp(x, batch)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = self.dropout(x)
        x1 = gmp(x, batch)
        C2ER = torch.reshape(C2ER, (1, 4))
        code = torch.cat([x0, x1, C2ER], dim=1)
        x = F.relu(self.lin1(code))
        x = F.relu(self.lin2(x))
        x = F.softmax(self.lin3(x), dim=1)  # batch个结果
        return x, code

#这里给出大家注释方便理解
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
        # data_list = get_data()
        data_list = []
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    KK = 2

    ShanghaiDataset = MyOwnDataset(f"Shanghai_K{KK}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net(K=KK).to(device)

    model.load_state_dict(torch.load(f'BC_K{KK}Epoch400T.pth'))
    model.eval()

    encoding = []

    for index, data in enumerate(ShanghaiDataset):
        print(f'---{index}---')
        data.to(device)
        x, shape_code = model(data)

        encoding.append(shape_code.reshape(-1).detach().cpu().numpy())

        _, pred = torch.max(x.data, 1)
        pred = pred.detach().cpu().numpy()[0]

    encoding = np.array(encoding)
    print(encoding.shape)

    np.savetxt(f"Shanghai_code_K{KK}.txt", encoding, fmt='%.8f')




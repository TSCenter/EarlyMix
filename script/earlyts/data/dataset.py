# coding=utf-8

import numpy as np
import os

from earlyts.config import *
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader




def load_dataset(dataset, batch_size, device, expand=True):

    label_id_index_dict = {}

    data_dir = os.path.join(data_root_dir, dataset)

    def read_data(type):
        X = []
        Y = []

        csv_path = os.path.join(data_dir, "{}_{}".format(dataset, type))
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                items = line.split(",")
                label_id = items[0]
                if label_id not in label_id_index_dict:
                    label_index = len(label_id_index_dict)
                    label_id_index_dict[label_id] = label_index
                else:
                    label_index = label_id_index_dict[label_id]
                features = [float(item) for item in items[1:]]
                X.append(features)
                Y.append(label_index)

        X = np.array(X).astype(np.float32)
        Y = np.array(Y).astype(np.int64)

        if expand:
            X = np.expand_dims(X, axis=-1)
        return X, Y

    train_x, train_y = read_data("TRAIN")
    test_x, test_y = read_data("TEST")

    def create_data_loader(x, y, shuffle):
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    train_data_loader = create_data_loader(train_x, train_y, shuffle=True)
    test_data_loader = create_data_loader(test_x, test_y, shuffle=False)



    return train_x, train_y, test_x, test_y, train_data_loader, test_data_loader
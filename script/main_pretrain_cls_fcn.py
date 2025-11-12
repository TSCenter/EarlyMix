# coding=utf-
# import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from tqdm import tqdm

from earlyts.data.dataset import load_dataset
from earlyts.logging.remote_logging import log
from earlyts.model.common import Lambda
import numpy as np
from argparse import ArgumentParser
import os
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from earlyts.model.mix_multi_fcn import FCN
import torchmetrics
import random

from earlyts.utils import reset_seed


parser = ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--num_epochs", default=2001, type=int)
parser.add_argument("--cls_type", type=str, default="variable", required=True)
parser.add_argument("--num_cnns", type=int, required=True)
parser.add_argument("--num_strides", type=int, default=50)
parser.add_argument("--model_base_dir", type=str, default="saved_cls_models")

parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--l2_coef", type=float, default=1e-5, help="l2 coef")

parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")

args = parser.parse_args()
print(args)
dataset = args.dataset
num_epochs = args.num_epochs
cls_type = args.cls_type
num_cnns = args.num_cnns
num_strides = args.num_strides
model_base_dir = args.model_base_dir
lr = args.lr
l2_coef = args.l2_coef
batch_size = args.batch_size
seed = args.seed
gpu_ids = args.gpu
device = "cuda"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

reset_seed(0)

if cls_type not in ["fixed", "variable"]:
    raise Exception("wrong loss type: ", cls_type)

# ==== create dir for saved model ====
model_dir = os.path.join(model_base_dir, "{}_{}_cnns_{}_num_strides_{}".format(dataset, cls_type, num_cnns, num_strides))
model_path = os.path.join(model_dir, dataset)
restore = os.path.exists(model_dir)
print("restore: ", restore)
if not restore:
    os.makedirs(model_dir)

# ==== load dataset (3-dimensional for X, [B, T, D]) and compute data-related parameters ====
train_x, train_y, test_x, test_y, train_data_loader, test_data_loader = load_dataset(dataset, batch_size=batch_size, device=device, expand=True)
num_classes = train_y.max() + 1

threshold = 0.98
len_ts = train_x.shape[1]
strides = int(np.ceil(len_ts / num_strides))
print("strides = {}".format(strides))

model = FCN(train_x.shape[-1], [64, 128], num_classes, num_cnns=num_cnns, stride=strides)
print(model(torch.tensor(train_x)).size())
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss(reduction="none")

metric_dict = {
    "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=int(num_classes)),
    "loss": torchmetrics.MeanMetric().to(device)
}

metric_dict = {
    metric_name: metric.to(device) for metric_name, metric in metric_dict.items()
}


def evaluate(data_loader):
    model.eval()
    if cls_type == "variable":
        accuracy_list = []
        loss_list = []
        for end in tqdm(np.arange(train_x.shape[1], 0, -strides * 5)):
            accuracy, loss = evaluate_once(data_loader, end)
            accuracy_list.append(accuracy)
            loss_list.append(loss)
        return np.mean(accuracy_list), np.mean(loss_list)
    else:
        return evaluate_once(data_loader)


def evaluate_once(data_loader, sub_ts_len=None):
    model.eval()
    with torch.no_grad():
        for metric in metric_dict.values():
            metric.reset()

        for batch_x, batch_y in data_loader:
            if sub_ts_len is not None:
                batch_x = batch_x[:, :sub_ts_len]
            logits = model(batch_x)
            preds = logits.argmax(dim=-1)

            metric_dict["accuracy"](preds, batch_y)
            losses = loss_func(logits, batch_y)
            metric_dict["loss"](losses)

        accuracy = metric_dict["accuracy"].compute().detach().cpu().numpy()
        cls_loss = metric_dict["loss"].compute().detach().cpu().numpy()

    return accuracy, cls_loss


best_accuracy = 0.0
best_cls_loss = 100000.0
best_epoch = None

for epoch in tqdm(range(num_epochs)):

    if epoch % 100 == 0:
        test_accuracy, test_cls_loss = evaluate(test_data_loader)
        print(test_cls_loss, best_cls_loss)
        if test_accuracy >= best_accuracy:
            if test_accuracy > best_accuracy or test_cls_loss < best_cls_loss:
                print("update best: {}({}) -> {}({})".format(best_accuracy, best_epoch, test_accuracy, epoch))
                best_accuracy = test_accuracy
                best_cls_loss = test_cls_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print("save model: ", test_accuracy)
        print("epoch = {}, accuracy = {}, cls_loss = {}".format(epoch, test_accuracy, test_cls_loss))
        print("best epoch = {}, best accuracy = {}, best cls_loss = {}".format(best_epoch, best_accuracy, best_cls_loss))


    for step, (batch_x, batch_y) in enumerate(train_data_loader):
        model.train()

        unique_labels = batch_y.unique()
        mixed_x_list = []
        mixed_y_list = []

        for label in unique_labels:
            indices = torch.where(batch_y == label)[0]
            class_samples = batch_x[indices] 
            x_mix = class_samples.mean(dim=0, keepdim=True)  
            mixed_x_list.append(x_mix)
            mixed_y_list.append(label.unsqueeze(0))

        if len(mixed_x_list) > 0:
            mixed_x = torch.cat(mixed_x_list, dim=0)
            mixed_y = torch.cat(mixed_y_list, dim=0)
            batch_x = torch.cat([batch_x, mixed_x], dim=0)
            batch_y = torch.cat([batch_y, mixed_y], dim=0)

        if cls_type == "variable":
            cls_losses_list = []
            for _ in range(4):
                start = torch.randint(0, strides, [])
                partial_len = torch.randint(1, train_x.shape[1], [])
                partial_batch_X = batch_x[:, start:start + partial_len]
                logits = model(partial_batch_X)
                cls_losses = loss_func(logits, batch_y)
                cls_losses_list.append(cls_losses)
            cls_loss = torch.stack(cls_losses_list, dim=0).mean()
        else:
            start = torch.randint(0, strides, [])
            logits = model(batch_x[:, start:])
            cls_losses = loss_func(logits, batch_y)
            cls_loss = cls_losses.mean()

        l2_loss = 0.0
        for name, param in model.named_parameters():
            if "weight" in name:
                l2_loss += (param ** 2).sum() * 0.5
        
        loss = cls_loss + l2_loss * l2_coef

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# log("{}[{},cnns_{},num_strides_{}]: accuracy={:.4f} epoch={}".format(dataset, cls_type, num_cnns, num_strides, best_accuracy, best_epoch))
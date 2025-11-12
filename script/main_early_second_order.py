# coding=utf-8

import os
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from tqdm import tqdm
from earlyts.data.dataset import load_dataset
from earlyts.model.mix_fcn import FCN
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from earlyts.model.likelihood import LikelihoodClassifierDecoder
from earlyts.utils import reset_seed


parser = ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--cls_type", type=str, required=True)
parser.add_argument("--likelihood_type", type=str, default='variable')
parser.add_argument("--num_cnns", type=int, required=True)

parser.add_argument("--num_strides", type=int, default=50)
parser.add_argument("--cls_model_base_dir", type=str, default="saved_cls_models")


parser.add_argument("--batch_size", type=int, default=1000, help="batch size")

parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")


args = parser.parse_args()
print(args)

dataset = args.dataset
num_epochs = args.num_epochs
num_cnns = args.num_cnns
num_strides = args.num_strides
cls_model_base_dir = args.cls_model_base_dir
batch_size = args.batch_size
gpu_ids = args.gpu
device = "cuda"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
reset_seed(0)

cls_type = args.cls_type
likelihood_type = args.likelihood_type

if cls_type not in ["fixed", "variable"]:
    raise Exception("wrong cls type: ", cls_type)

if likelihood_type not in ["fixed", "variable", "concat"]:
    raise Exception("wrong likelihood type: ", likelihood_type)

date_str = datetime.now().strftime("%Y-%d-%m %H:%M:%S")

with open("outputs/{}.log".format(dataset), "a", encoding="utf-8") as f:
    f.write("\n==== {} ====\n".format(date_str))


train_x, train_y, test_x, test_y, train_raw_data_loader, test_raw_data_loader = load_dataset(dataset, batch_size=batch_size, device=device, expand=True)
num_classes = train_y.max() + 1
augmentation = train_x.shape[0] < 100



drop_rate = 0.3
likelihood_drop_rate = drop_rate
att_drop_rate = 0.0

lr = 1e-3
l2_coef = 5e-4
likelihood_l2_coef = l2_coef

units = 256

num_transformers = 3
thresholds = np.linspace(0.0, 1.0, 101)

noise_coef = None  # 0.1 if augmentation else None
heads = 4

sampling_internal = None
warmup_steps = 200
# strides = 1
len_ts = train_x.shape[1]
strides = int(np.ceil(len_ts / num_strides))
print("strides: ", strides)

likelihood_units = units
likelihood_heads = heads
early_threshold = 0.95  # 0.6

print(f"units: {units}\theads: {heads}\tinterval: {sampling_internal}"
      f"\tdrop_rate: {drop_rate}"
      f"\tl2_coef: {l2_coef}"
      f"\tnoisy_coef: {noise_coef}\tatt_drop_rate: {att_drop_rate}")



# metric_dict = {
#     "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=int(num_classes)),
#     "loss": torchmetrics.MeanMetric().to(device)
# }

full_cls_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=int(num_classes)).to(device)



def evaluate_full_cls_model(cls_model, data_loader):
    cls_model.eval()

    with torch.no_grad():

        full_cls_accuracy_metric.reset()

        for batch_x, batch_y in data_loader:
            batch_logits = cls_model(batch_x)
            batch_preds = torch.argmax(batch_logits, dim=-1)
            full_cls_accuracy_metric(batch_preds, batch_y)

        accuracy = full_cls_accuracy_metric.compute().detach().cpu().item()
    
    return accuracy


def load_cls_model(cls_type):
    cls_model = FCN(train_x.shape[-1], [64, 128], num_classes, num_cnns=num_cnns, stride=strides)
    cls_model_dir = os.path.join(cls_model_base_dir, "{}_{}_cnns_{}_num_strides_{}".format(dataset, cls_type, num_cnns, num_strides))
    cls_model_path = os.path.join(cls_model_dir, dataset)
    cls_model.load_state_dict(torch.load(cls_model_path))

    cls_model = cls_model.to(device)

    print("restore ....")
    print("init accuracy[{}]: ".format(cls_type), evaluate_full_cls_model(cls_model, test_raw_data_loader))
    return cls_model



cls_model = load_cls_model("variable")


def build_likelihood_dataset(raw_data_loader, batch_size, shuffle):
    cls_model.eval()
    with torch.no_grad():

        batch_probs_list = []
        batch_corrects_list = []

        for batch_x, batch_y in tqdm(raw_data_loader):
            batch_sub_probs_list = []
            batch_sub_corrects_list = []
            for sub_len in range(1, len_ts + 1, strides):
                batch_sub_x = batch_x[:, :sub_len]
                batch_sub_logits =  cls_model(batch_sub_x)
                batch_sub_probs = F.softmax(batch_sub_logits, dim=-1)
                batch_sub_probs_list.append(batch_sub_probs)

                batch_sub_preds = batch_sub_logits.argmax(dim=-1)
                batch_sub_corrects = (batch_sub_preds == batch_y).float()
                batch_sub_corrects_list.append(batch_sub_corrects)


            batch_probs = torch.stack(batch_sub_probs_list, dim=1)
            batch_corrects = torch.stack(batch_sub_corrects_list, dim=1)
            batch_probs_list.append(batch_probs)
            batch_corrects_list.append(batch_corrects)

        probs = torch.cat(batch_probs_list, dim=0)
        corrects = torch.cat(batch_corrects_list, dim=0)

        likelihood_x = probs
        likelihood_y = corrects
        likelihood_dataset = TensorDataset(likelihood_x, likelihood_y)
        likelihood_data_loader = DataLoader(likelihood_dataset, batch_size=batch_size, shuffle=shuffle)

    return likelihood_x.detach().cpu().numpy(), likelihood_y.detach().cpu().numpy(), likelihood_data_loader


_, _, train_data_loader = build_likelihood_dataset(train_raw_data_loader, batch_size, shuffle=True)
_, _, test_data_loader = build_likelihood_dataset(test_raw_data_loader, batch_size, shuffle=False)



likelihood_model = LikelihoodClassifierDecoder(num_classes, likelihood_units,
                                               len_ts, strides=strides, heads=likelihood_heads,
                                               drop_rate=likelihood_drop_rate).to(device)

loss_func = nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.Adam(likelihood_model.parameters(), lr=lr)





def predict_likelihood(data_loader):
    likelihood_model.eval()
    with torch.no_grad():
        batch_likelihood_probs_list = []
        batch_combined_likelihood_probs_list = []
        batch_likelihood_y_list = []
        for batch_likelihood_x, batch_likelihood_y in data_loader:
            batch_likelihood_logits = likelihood_model(batch_likelihood_x)
            batch_likelihood_probs = F.sigmoid(batch_likelihood_logits).squeeze(-1)

            batch_cls_max_probs = batch_likelihood_x.max(dim=-1)[0]
            batch_combined_likelihood_probs =  batch_likelihood_probs * batch_cls_max_probs

            batch_likelihood_probs_list.append(batch_likelihood_probs)
            batch_likelihood_y_list.append(batch_likelihood_y)
            batch_combined_likelihood_probs_list.append(batch_combined_likelihood_probs)




        likelihold_probs = torch.cat(batch_likelihood_probs_list, dim=0).detach().cpu().numpy()
        combined_likelihood_probs = torch.cat(batch_combined_likelihood_probs_list, dim=0).detach().cpu().numpy()
        likelihood_y = torch.cat(batch_likelihood_y_list, dim=0).detach().cpu().numpy()

    return likelihold_probs, combined_likelihood_probs, likelihood_y



def predict_early(combined_likelihood_probs, likelihood_y, current_early_threshold, report=False):

    likelihood_preds_matrix = (combined_likelihood_probs >= current_early_threshold).astype(np.int32)


    early_times = []
    early_corrects = []

    for test_index, (likelihood_preds, cls_corrects) in enumerate(zip(likelihood_preds_matrix, likelihood_y)):
        early_correct = cls_corrects[-1]
        early_time = len_ts

        for time_index, (likelihood_pred, cls_correct) in enumerate(zip(likelihood_preds, cls_corrects)):
            if likelihood_pred:
                early_correct = cls_correct
                early_time = (time_index + 1) * strides
                break

        early_corrects.append(early_correct)
        early_times.append(early_time)

    early_accuracy = np.array(early_corrects).mean()
    early_rate = np.mean(np.array(early_times) / len_ts)
    earlyness = early_accuracy / early_rate
    hm = 2*early_accuracy*(1-early_rate) / (early_accuracy + (1-early_rate))
    return early_accuracy, early_rate, earlyness, hm
  









class Result(object):
    def __init__(self, early_accuracy, early_rate, earlyness, hm, epoch=None):
        self.early_accuracy = early_accuracy
        self.early_rate = early_rate
        self.earlyness = earlyness
        self.epoch = epoch
        self.hm = hm
    def __str__(self):
        return "epoch = {}\tearly_accuracy = {}\tearly_rate = {}\thm = {}".format(
            self.epoch, self.early_accuracy, self.early_rate, self.hm
        )


best_result = None

loss = None
cls_loss = None
l2_loss = None
mask_rate = None
use_likelihood = True

for epoch in tqdm(range(num_epochs)):

    if epoch % 100 == 0:

        train_likelihold_probs, train_combined_likelihood_probs, train_likelihood_y = predict_likelihood(train_data_loader)

        threshold_X = train_combined_likelihood_probs.reshape(-1, 1)
        threshold_y = train_likelihood_y.flatten()

        cls = LogisticRegression(class_weight="balanced")
        cls.fit(threshold_X, threshold_y)
        threshold_labels = cls.predict(thresholds.reshape(-1, 1))
        early_threshold = 0.6
        for threshold, threshold_label in zip(thresholds, threshold_labels):
            if threshold_label == 1:
                early_threshold = threshold
                break



     

        test_likelihold_probs, test_combined_likelihood_probs, test_likelihood_y = predict_likelihood(test_data_loader)




        for threshold in thresholds:
            early_accuracy, early_rate, earlyness, hm = predict_early(test_combined_likelihood_probs, test_likelihood_y, threshold)
            print(
                "epoch = {}\tthreshold = {}\tearly_accuracy = {:.4f}\tearly_rate = {:.4f}\tearlyness = {:.4f}\thm = {:.4f}"
                .format(epoch, threshold, early_accuracy, early_rate, earlyness, hm))

        print("early threshold = {}".format(early_threshold))
        early_accuracy, early_rate, earlyness, hm = predict_early(test_combined_likelihood_probs, test_likelihood_y, early_threshold, report=True)

        log_str = "epoch = {}\tearly_accuracy = {}\tearly_rate = {}\tearlyness = {}\thm = {}" \
            .format(epoch, early_accuracy, early_rate, earlyness, hm)
        print(log_str)

        best_updated = False
        if best_result is None or best_result.earlyness < earlyness or best_result.early_accuracy < early_accuracy - 0.02:
            hm = 2*early_accuracy*(1-early_rate) / (early_accuracy + (1-early_rate))
            best_result = Result(early_accuracy, early_rate, earlyness, hm, epoch=epoch)
            best_updated = True
        print("[{}]Best Result: {}".format("new" if best_updated else "old", best_result))
        print()

        with open("outputs/{}.log".format(dataset), "a", encoding="utf-8") as f:
            f.write("{}\n".format(args))
            f.write("{}\n".format(log_str))

    
    for step, (batch_x, batch_y) in enumerate(train_data_loader):
        likelihood_model.train()
        logits = likelihood_model(batch_x).squeeze(dim=-1)
        likelihood_cls_losses = loss_func(logits, batch_y)
        likelihood_cls_loss = likelihood_cls_losses.mean()

        l2_loss = 0.0

        for name, param in likelihood_model.named_parameters():
            if "weight" in name:
                l2_loss += (param ** 2).sum() * 0.5
        
        loss = likelihood_cls_loss + l2_loss * l2_coef

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



with open("outputs/all.log", "a", encoding="utf-8") as f:
    f.write("{}[{}, {}]: {}\n".format(dataset, cls_type, likelihood_type, best_result))

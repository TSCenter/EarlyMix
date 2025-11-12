# coding=utf-8
import random
import torch
import numpy as np

def reset_seed(seed):
    print("set seed: {} ...".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

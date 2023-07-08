import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

import argparse
from add_columns import load_mm
import numpy as np
import json
from tqdm import tqdm
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

import time
import pynvml   # 显存

batch_size =10000

start = time.perf_counter() #开始时间

p = argparse.ArgumentParser()

p.add_argument('--workers', type = int, default=32)
p.add_argument('--image_input_dir', type = str, help='Directory to input image')
p.add_argument('--image_type', type = str, help='image_type')
p.add_argument('--json_input_dir', type = str, help='Directory to input image')
p.add_argument('--output_dir', type = str, help='Directory to save result to')
p.add_argument('--batch_size', type = int, default = batch_size, help='Directory to save result to')
p.add_argument('--model_path', type = str, help='Directory to model')
p.add_argument('--shard_id', type = int, default=0, help='shard id')
p.add_argument('--shard_size', type = int, default=0, help='shard id')
args = p.parse_args()
#####  This script will predict the aesthetic score for this image file:

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load(
    f"{args.model_path}/sac+logos+ava1-l14-linearMSE.pth")  # load the model you trained previously or the model available in this repo

model.load_state_dict(s)
model.to("cuda")
model.eval()

# print("Aesthetic score predicted by the model:")
# print(prediction)
# model = nn.Linear(512, 1)
# s = torch.load(f"{args.model_path}/sa_0_4_vit_b_32_linear.pth")
# model.load_state_dict(s)
# model.eval()

all_pairs = load_mm(os.path.join(args.json_input_dir, f"{args.image_type}_safety_{args.shard_id}.json"))
np_emb = np.memmap(f"{args.image_input_dir}/{args.image_type}_{args.shard_id}.memmap", mode="r", dtype=np.float16).reshape(-1, 768)
# with open(f"{args.image_input_dir}/{args.image_type}_{args.shard_id}.json", 'r') as f:
#     image_dict = json.load(f)

np_bias = 0
all_image_cnt = len(all_pairs)
new_pairs = []
files_cache = []
idx_cache = []
for name, pair in tqdm(all_pairs.items()):
    files_cache.append(pair)
    idx_cache.append(pair['emb_idx'])
    cache_emb = torch.from_numpy(normalized(np_emb[idx_cache])).cuda().float()
    cache_scores = model(cache_emb).cpu().detach().numpy().tolist() # 计算美学分数
    for bias in range(cache_emb.shape[0]):
        if cache_scores[bias][0] / 100.0 > 0.04:
            pair = files_cache[bias]
            pair['aesthetics_score'] = cache_scores[bias][0] / 100.0
            new_pairs.append(pair)
            files_cache = []
            idx_cache = []

print(f"filter_image:{len(new_pairs)}")
with open(f"{args.image_input_dir}/{args.image_type}_safety_aesthetics_{args.shard_id}.json", 'w') as f:
    json.dump(new_pairs, f, ensure_ascii=True)
    
    
end = time.perf_counter() #结束时间


# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("batch size::", batch_size)
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")





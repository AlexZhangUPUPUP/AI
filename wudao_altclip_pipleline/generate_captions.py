import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
import asyncio
from asyncio_buffered_pipeline import buffered_pipeline
import argparse
from tqdm import tqdm

p = argparse.ArgumentParser()

p.add_argument('--workers', type = int, default=128)
p.add_argument('--image_input_dir', type = str, help='Directory to input image')
p.add_argument('--image_type', type = str, help='image_type')
p.add_argument('--json_input_dir', type = str, help='Directory to input image')
p.add_argument('--output_dir', type = str, help='Directory to save result to')
p.add_argument('--batch_size', type = int, default = 32, help='Directory to save result to')
p.add_argument('--shard_id', type = int, default=0, help='shard id')
p.add_argument('--shard_size', type = int, default=0, help='shard id')
args = p.parse_args()

print("Creating model")
model = blip_decoder(pretrained='model_base_caption_capfilt_large.pth', image_size=384, vit='base', 
                        vit_grad_ckpt=False, vit_ckpt_layer=0, 
                        prompt= 'a picture of ')

if torch.cuda.is_available():
#     model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to("cuda:0")
print("setup transforms")

def load_mm(input_path, access_way="imagePath", list_first=False):
    allpairs = {}
    with open(input_path, 'r') as f:
        batchpairs = json.load(f)
        if not list_first:
            batchpairs = batchpairs['RECORDS']
        for pair in tqdm(batchpairs):
            key = pair[access_way].split("/")[-1]
            allpairs[key] = pair
    return allpairs

print(" load images ")
import json
wudao_art_path = os.path.join(f'{args.json_input_dir}',f'{args.image_type}_safety_aesthetics_watermark_{args.shard_id}.json')
allpairs = load_mm(wudao_art_path, list_first=True)
new_pairs= []
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
preprocess = transforms.Compose([
                transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
                ])

async def image_loader(input_path):
    image_batches = []
    files = []
    for name, pair, in tqdm(allpairs.items()):
        try:
            image_batches.append(Image.open(os.path.join(input_path, name)).convert('RGB'))
            files.append(pair)
        except:
            continue
        if len(image_batches) % args.batch_size == 0:
            yield (files, image_batches)
            files = []
            image_batches = []
    if image_batches:
        yield (files, image_batches)

async def image_preprocess(image_batch_iter):
    async for image_batch in image_batch_iter:
        files, image_batches = image_batch
        image_tensor_batch = []
        new_files = []
        for idx, image in enumerate(image_batches):
            try:
                image_tensor_batch.append(preprocess(image).unsqueeze(0))
                new_files.append(files[idx])
            except:
                continue
        yield (new_files, image_tensor_batch)

async def image_infer(image_tensor_iter):
    global new_pairs
    async for image_tensor in image_tensor_iter:
        files, image_tensor = image_tensor
        image_tensor = torch.cat(image_tensor, dim=0).to("cuda:0")
        captions = model.generate(image_tensor, sample=False, num_beams=1, max_length=20,
                                  min_length=5)
        for fi, fn in enumerate(captions):
            files[fi]['caption_blip'] = captions[fi]
        new_pairs += files

buffer_iterable = buffered_pipeline()
read_it = buffer_iterable(image_loader(f"{args.image_input_dir}/{args.image_type}"), buffer_size=7)
preprocess_it = buffer_iterable(image_preprocess(read_it), buffer_size=7)
asyncio.run(image_infer(preprocess_it))

with open(f"{args.output_dir}/{args.image_type}_safety_aesthetics_watermark_caption_{args.shard_id}.json", 'w') as f:
    json.dump(new_pairs, f, ensure_ascii=False)

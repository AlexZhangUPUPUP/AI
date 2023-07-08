import argparse

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
import os
import argparse
from add_columns import load_mm
import numpy as np
import json
from tqdm import tqdm
import asyncio
from asyncio_buffered_pipeline import buffered_pipeline


import time
import pynvml   # 显存

# batch_size =250

start = time.perf_counter() #开始时间


p = argparse.ArgumentParser()

p.add_argument('--workers', type = int, default=128)
p.add_argument('--image_input_dir', type = str, help='Directory to input image')
p.add_argument('--image_type', type = str, help='image_type')
p.add_argument('--json_input_dir', type = str, help='Directory to input image')
p.add_argument('--output_dir', type = str, help='Directory to save result to')
p.add_argument('--batch_size', type = int, default = 250, help='Directory to save result to')
p.add_argument('--shard_id', type = int, default=0, help='shard id')
p.add_argument('--shard_size', type = int, default=0, help='shard id')

args = p.parse_args()
args.batch_size=270

preprocessing = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

preprocessing_sub = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

if __name__ == '__main__':

    model = timm.create_model(
        'efficientnet_b3a', pretrained=True, num_classes=2)

    model.classifier = nn.Sequential(
        # 1536 is the orginal in_features
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    state_dict = torch.load('models/watermark_model_v1.pt')

    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda:0")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    all_pairs = load_mm(os.path.join(args.json_input_dir, f"{args.image_type}_safety_aesthetics_{args.shard_id}.json"))
    new_pairs = []

    async def image_loader(input_path):
        image_batches = []
        files = []
        # with open(input_path, 'r') as f:
        #     image_dict = json.load(f)
        for name, pair in tqdm(all_pairs.items()):
            # if fi < args.shard_id * args.shard_size:
            #     continue
            # elif fi >= (args.shard_id + 1) * args.shard_size:
            #     break
            try:
                image_batches.append(Image.open(os.path.join(f"{args.image_input_dir}/{args.image_type}", name)))
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
            filter_files = []
            image_tensor_batch = []
            for idx, image in enumerate(image_batches):
                try:
                    image_tensor_batch.append(preprocessing(image).unsqueeze(0))
                    filter_files.append(files[idx])
                except:
                    continue
            yield (filter_files, image_tensor_batch)


    async def image_infer(image_tensor_iter):
        async for image_tensor in image_tensor_iter:
            files, image_tensor = image_tensor
            image_emb = torch.cat(image_tensor, dim=0).to("cuda:0")
            watermark_score = F.softmax(model(image_emb), dim=1)[:,1].cpu().detach().numpy().tolist()
            for idx, image_id in enumerate(files):
                if watermark_score[idx] > 0.2:
                    pair = files[idx]
                    pair['clean_score'] = watermark_score[idx]
                    new_pairs.append(pair)


    buffer_iterable = buffered_pipeline()
    read_it = buffer_iterable(image_loader(f"{args.json_input_dir}/{args.image_type}_{args.shard_id}.json"), buffer_size=10)
    preprocess_it = buffer_iterable(image_preprocess(read_it), buffer_size=10)
    asyncio.run(image_infer(preprocess_it))

    with open(f"{args.output_dir}/{args.image_type}_safety_aesthetics_watermark_{args.shard_id}.json", 'w') as f:
        json.dump(new_pairs, f, ensure_ascii=True)


end = time.perf_counter() #结束时间

# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("batch size::", batch_size)
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")



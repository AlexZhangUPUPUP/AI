from PIL import Image
import requests
from transformers import AutoProcessor, AltCLIPModel
from transformers import AltCLIPModel, AltCLIPProcessor
import torch
import torch.nn.functional as F
import time

from add_columns import load_mm
import argparse
import os
import multiprocess
import json

from tqdm import tqdm
import numpy as np

import asyncio
from asyncio_buffered_pipeline import buffered_pipeline


# 开始时间
time1 = time.time()
path_folder ="/share/projset/baaishare/baai-mmdataset/wudaomm-5m"


model = AltCLIPModel.from_pretrained("/home/alex/Alt_clip_Wudao/model/checkpoint-258420").to("cuda")
processor = AltCLIPProcessor.from_pretrained("/home/alex/Alt_clip_Wudao/model/checkpoint-258420")
print("Point0")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


inputs = processor(text=["一只猫和一只狗"], images=image, return_tensors="pt", padding=True).to("cuda")


# "a couple of cats laying on top of a pink couch, 2006 photograph, wisps of energy in the air, bumpy mottled skin, 2 d sprites, hazard stripes, dressed in a frilly ((ragged)), 
# sleepers, mute dramatic colours, twitch tv, size difference, maintenance photo, vignette"
outputs = model(**inputs) # 包含embadding 概率  similarity score
print("Point1")

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
logits_per_text  = outputs.logits_per_text   # 

text_embeds = outputs.text_embeds  # embadding of text
image_embeds = outputs.image_embeds # embadding of image

probs = logits_per_image.softmax(dim=1)  # 进过softmax的概率

cos_sim = torch.cosine_similarity(image_embeds,text_embeds).detach().cpu().numpy() # 余弦相似度
print("sim "+str(cos_sim))

time2 = time.time()

print('time cost: %.6f'%(time2-time1))



# end = time.perf_counter() #结束时间


# # 计算运行时间
# runTime = end - start
# runTime_ms = runTime * 1000
# print("运行时间：", runTime, "秒")
# print("运行时间：", runTime_ms, "毫秒")

import clip
from PIL import Image
from add_columns import load_mm
import argparse
import os
import multiprocess
import json
import time
from tqdm import tqdm
import numpy as np
import torch
import asyncio
from asyncio_buffered_pipeline import buffered_pipeline
import time
import pynvml   # 显存

batch_size =3500


start = time.perf_counter() #开始时间

model, preprocess = clip.load('ViT-L/14')
visual_model = model.visual.to("cuda:0")
visual_model = torch.nn.DataParallel(visual_model, device_ids=list(range(torch.cuda.device_count())))

p = argparse.ArgumentParser()
p.add_argument('--workers', type = int, default=1024)
p.add_argument('--image_input_dir', type = str, help='Directory to input image')
p.add_argument('--image_type', type = str, help='image_type')
p.add_argument('--json_input_dir', type = str, help='Directory to input image')
p.add_argument('--output_dir', type = str, help='Directory to save result to')
p.add_argument('--batch_size', type = int, default=batch_size, help='Directory to save result to') #默认64
p.add_argument('--shard_id', type = int, default=0, help='shard id')
p.add_argument('--shard_size', type = int, default=500000, help='shard id')
args = p.parse_args()
args.batch_size = args.batch_size * torch.cuda.device_count()
all_pairs = load_mm(os.path.join(args.json_input_dir, f"{args.image_type}.json"))
image_cnt = 0
image_dict = []

# 申请内存
image_embs = np.memmap(os.path.join(args.output_dir, f"{args.image_type}_{args.shard_id}.memmap"), shape=(args.shard_size, 768), mode='w+', dtype=np.float16)

def image_loader(input_path):
    image_tensor_batch = []
    files = []
    for fi, file in tqdm(enumerate(os.listdir(input_path))):
        if fi < args.shard_id * args.shard_size:
            continue
        elif fi >= (args.shard_id + 1) * args.shard_size:
            break
        try:
            image = Image.open(os.path.join(input_path, file))
            image_tensor_batch.append(preprocess(image).unsqueeze(0))
            files.append(file)
            if len(image_tensor_batch) % args.batch_size == 0:
                yield (files, image_tensor_batch)
                files = []
                image_tensor_batch = []
        except:
            continue
    if image_tensor_batch:
        yield (files, image_tensor_batch)

# def image_preprocess(image_batch_iter):
#     for files, image_batches in image_batch_iter:
#         image_tensor_batch = []
#         for image in image_batches:
#
#         yield (files, image_tensor_batch)
#
# buffer_iterable = buffered_pipeline()
# read_it = buffer_iterable(image_loader(f"{args.image_input_dir}/{args.image_type}"), buffer_size=7)


embadding_start = time.perf_counter() #embadding开始时间



pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 指定显卡号
print("\n\n总的显存大小1")
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.total/1024**3) #总的显存大小（float）

print("已用显存大小1")
print(meminfo.used/1024**3)  #已用显存大小（float）

print("剩余显存大小1")
print(meminfo.free/1024**3)  #剩余显存大小（float）
print()

# embadding
with torch.no_grad(): # 不需要计算梯度
    for image_tensor in image_loader(f"{args.image_input_dir}/{args.image_type}"):
            files, image_tensor = image_tensor
            image_emb = torch.cat(image_tensor, dim=0).half().to("cuda:0")
            image_emb = visual_model(image_emb).cpu().detach().numpy()
            image_embs[image_cnt: image_cnt + image_emb.shape[0]] = image_emb # 写入memmap
            image_dict += files
            image_cnt += image_emb.shape[0]
            
            
            # print()
            # allocated = torch.cuda.max_memory_allocated(device=None)
            # # allocated = torch.cuda.memory_cached(device=None) # 剩余显存
            # print("torch 使用显存:    ",allocated)
            
            # rest = torch.cuda.memory_reserved(0)
            # # rest = torch.cuda.empty_cache()  剩余缓存
            # print("torch 剩余显存:    ",rest)

        
        
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 指定显卡号
            print("\n\n总的显存大小1")
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(meminfo.total/1024**3) #总的显存大小（float）

            print("已用显存大小1")
            print(meminfo.used/1024**3)  #已用显存大小（float）

            print("剩余显存大小1")
            print(meminfo.free/1024**3)  #剩余显存大小（float）
            print()
        

# preprocess_it = buffer_iterable(image_preprocess(read_it), buffer_size=7)
# asyncio.run(image_infer(preprocess_it))

embadding_end = time.perf_counter() #embadding 结束时间


# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 指定显卡号
# print("\n\n总的显存大小2")
# meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# print(meminfo.total/1024**3) #总的显存大小（float）
# print("已用显存大小2")
# print(meminfo.used/1024**3)  #已用显存大小（float）
# print("剩余显存大小2")
# print(meminfo.free/1024**3)  #剩余显存大小（float）
# print()

# 写json
image_dict = {value:key for key,value in enumerate(image_dict)}
with open(os.path.join(args.output_dir, f"{args.image_type}_{args.shard_id}.json"), 'w') as f:
    json.dump(image_dict, f, ensure_ascii=True)
    
    
end = time.perf_counter() #结束时间


# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")


embadding_time = embadding_end - embadding_start
embadding_time_ms =  embadding_time *1000

# 输出运行时间
print("embadding运行时间：", embadding_time, "秒")
print("embadding运行时间：", embadding_time_ms, "毫秒")
print("batch size::", batch_size)

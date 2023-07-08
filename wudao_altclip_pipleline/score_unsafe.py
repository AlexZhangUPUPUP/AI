import argparse
from add_columns import load_mm
import numpy as np
import json
from tqdm import tqdm
import os
import time
import time
import pynvml   # 显存
p = argparse.ArgumentParser()

batch_size =20000
start = time.perf_counter() #开始时间

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

def load_safety_model():
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel

    model_dir = f"{args.model_path}/clip_autokeras_binary_nsfw"
    dim = 768

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS, compile=False)
    loaded_model.predict(np.random.rand(10 ** 3, dim).astype("float32"), batch_size=10 ** 3)

    return loaded_model

all_pairs = load_mm(os.path.join(args.image_input_dir, f"{args.image_type}.json"))
safety_model = load_safety_model()
np_emb = np.memmap(f"{args.json_input_dir}/{args.image_type}_{args.shard_id}.memmap", mode="r", dtype=np.float16).reshape(-1, 768)
with open(f"{args.json_input_dir}/{args.image_type}_{args.shard_id}.json", 'r') as f:
    image_dict = json.load(f)




all_image_cnt = len(image_dict)
# image_dict = {value:key for value, key in image_dict.items()}
np_bias = 0
files_cache = []
idx_cache= []
new_pairs = []
for file, idx in tqdm(image_dict.items()):
    if file not in all_pairs:
        continue
    files_cache.append(file)
    idx_cache.append(idx + np_bias)
    if len(files_cache) == args.batch_size:
        cache_emb = np_emb[idx_cache]
        cache_scores = safety_model.predict(cache_emb, batch_size=cache_emb.shape[0]).tolist()
        for bias in range(cache_emb.shape[0]):
            if cache_scores[bias][0] > 0.5:
                pair = all_pairs[files_cache[bias]]
                pair['safety_score'] = cache_scores[bias][0]
                pair['emb_idx'] = idx_cache[bias]
                new_pairs.append(pair)
        files_cache = []
        idx_cache = []
        
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 指定显卡号
        # print("\n\n总的显存大小1")
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.total/1024**3) #总的显存大小（float）

        # print("已用显存大小1")
        # print(meminfo.used/1024**3)  #已用显存大小（float）

        # print("剩余显存大小1")
        # print(meminfo.free/1024**3)  #剩余显存大小（float）
        # print()
    
        

if files_cache:
    cache_emb = np_emb[idx_cache]
    cache_scores = safety_model.predict(cache_emb, batch_size=cache_emb.shape[0]).tolist()
    for bias in range(cache_emb.shape[0]):
        if True or cache_scores[bias][0] > 0.5:
            pair = all_pairs[files_cache[bias]]
            pair['safety_score'] = cache_scores[bias][0]
            pair['emb_idx'] = idx_cache[bias]
            new_pairs.append(pair)
    files_cache = []
    idx_cache = []
print(f"new length of all samples:{len(new_pairs)}")
with open(f"{args.json_input_dir}/{args.image_type}_safety_{args.shard_id}.json", 'w') as f:
    json.dump(new_pairs, f, ensure_ascii=True)


end = time.perf_counter() #结束时间

# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")
print("batch size::", batch_size)
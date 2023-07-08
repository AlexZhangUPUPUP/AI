import json
from tqdm import tqdm
import os
import time
i = 0
all_datas = []
with open("/sharefs/baai-mmdataset/wudaomm-inter/face/images_safety_aesthetics_0.json", 'r') as f:
    allpairs = json.load(f)
for pair in tqdm(allpairs):
    if 'aesthetics_score' not in pair:
        #print(pair)
        #time.sleep(10)
        continue
    all_datas.append(pair['aesthetics_score']*100)
    if i < 100 and pair['aesthetics_score'] > 0.055:
        name = pair['imagePath'].split("/")[-1]
        os.system(f"cp /sharefs/webbrain-lijijie/resources/data/images/{name} /sharefs/webbrain-lijijie/beauty_pics/{name}")
        i += 1

import pandas as pa

buckets = [i/4 for i in range(40)]
res = pa.cut(all_datas, buckets)
print(pa.value_counts(res))



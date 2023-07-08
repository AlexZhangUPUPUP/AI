import json
import os
import multiprocessing  as mlp
from tqdm import tqdm
input_dir = "/sharefs/baai-mmdataset/wudaomm-5m"
output_dir = "/sharefs/baai-mmdataset/wudaomm-5m-preprocessed"
preprocessed_num = 0
match_dict = {}

for sfile in os.listdir(input_dir):
    if sfile.endswith(".json") and sfile.startswith("Travel"):
        with open(os.path.join(input_dir, sfile), 'r') as f:
            allpairs = json.load(f)
            for pair in tqdm(allpairs):
                pair['id'] = f"wudao_{preprocessed_num}"
                preprocessed_num += 1
        #with open(os.path.join(output_dir, sfile), 'w') as wf:
            #json.dump(allpairs, wf, ensure_ascii=False)


        


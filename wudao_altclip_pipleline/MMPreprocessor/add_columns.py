import json
import os
from tqdm import tqdm
input_dir = "/sharefs/baai-mmdataset/wudaomm-5m"

def load_mm(input_path, access_way="name"):
    allpairs = {}
    with open(input_path, 'r') as f:
        batchpairs = json.load(f)
        for pair in tqdm(batchpairs):
            allpairs[pair[access_way]] = pair
    return allpairs

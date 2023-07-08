import json
import os
from tqdm import tqdm
import time
input_dir = "/sharefs/baai-mmdataset/wudaomm-5m"

def load_mm(input_path, access_way="imagePath", list_first=False):
    allpairs = {}
    with open(input_path, 'r') as f:
        batchpairs = json.load(f)
        if not isinstance(batchpairs, list):
            batchpairs = batchpairs['RECORDS']
        if 'name' in batchpairs[0]:
            access_way = 'name'
        else:
            access_way = "imagePath"
        for pair in tqdm(batchpairs):
            key = pair[access_way].split("/")[-1]
            allpairs[key] = pair
    return allpairs

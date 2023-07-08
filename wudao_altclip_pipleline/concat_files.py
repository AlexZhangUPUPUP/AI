import json
from tqdm import tqdm
all_json_files = []

for json_idx in tqdm(range(1, 2001)):
    json_idx_str = str(json_idx + 1000000)[1:]
    json_name = f"/sharefs/baai-mmdataset/diffusiondb/part-{json_idx_str}/part-{json_idx_str}.json"
    with open(json_name, 'r') as f:
        all_records = json.load(f)
        for key in all_records:
            json_obj = {"name":f"{json_idx_str}-{key}", "meta":all_records[key]}
            all_json_files.append(json_obj)

with open("/sharefs/baai-mmdataset/diffusiondb/images.json",'w') as f:
    json.dump(all_json_files, f)

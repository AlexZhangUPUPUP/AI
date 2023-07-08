import json
import glob
import numpy as np
import os
#替换成自己的不同的域名以访问不同域的图片
domain = "Art"
output_json_path = "/sharefs/baai-mmdataset/wudaomm-inter/wudaomm-5m"
image_path = "/sharefs/baai-mmdataset/wudaomm-5m"

#取过滤后的图片路径
#with open(f"{output_json_path}/{domain}_safety_aesthetics_watermark_0.json") as f:
 #   filter_image_dict = json.load(f)
  #  for image_item in filter_image_dict:
   #     image_name = image_item['name']
   #     print(f"{image_path}/{domain}/{image_name}")
            
#取openclip infer的、未过滤的全部图片embedding，极少数图片可能因为无法预处理没有编码
image_embs = np.memmap(os.path.join(output_json_path, f"{domain}_0.memmap"), shape=(500000, 768), mode='r', dtype=np.float16)
with open((f"{output_json_path}/{domain}_0.json"), 'r') as f:
    filter_image_dict = json.load(f)
    for image_name, emb_idx in filter_image_dict.items():
        print(image_embs[emb_idx])

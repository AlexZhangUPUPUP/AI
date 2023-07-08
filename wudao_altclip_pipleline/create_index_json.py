import json
from tqdm import tqdm

imagelist = []
imagetype = ""
with open("/sharefs/webbrain-lijijie/resources/data/caption.txt", 'r') as f:
    for li, line in tqdm(enumerate(f)):
        image_name, caption = line.split("\t")
        caption = caption.replace("\u0001","，")
        image_instance = {
            "id": str(li),
            "imageType": "人脸",
            "cnShortText": caption,
            "imagePath": f"./imagesFolder/{image_name}",
            "crawlTime": "2022-10-27 18:00:00"
        }
        imagelist.append(image_instance)

image_obj = {"RECORDS":imagelist}

with open("/sharefs/webbrain-lijijie/resources/data/images.json", 'w') as wf:
    json_str = json.dump(image_obj, wf, ensure_ascii=False)
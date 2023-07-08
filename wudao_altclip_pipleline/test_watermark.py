import argparse

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
import os
import argparse
from add_columns import load_mm
import numpy as np
import json
from tqdm import tqdm
p = argparse.ArgumentParser()

p.add_argument('--workers', type = int, default=32)
p.add_argument('--image_input_dir', type = str, help='Directory to input image')
p.add_argument('--image_type', type = str, help='image_type')
p.add_argument('--json_input_dir', type = str, help='Directory to input image')
p.add_argument('--output_dir', type = str, help='Directory to save result to')
p.add_argument('--batch_size', type = int, default = 128, help='Directory to save result to')
args = p.parse_args()
preprocessing = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

preprocessing_sub = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

if __name__ == '__main__':
    model = timm.create_model(
        'efficientnet_b3a', pretrained=True, num_classes=2)

    model.classifier = nn.Sequential(
        # 1536 is the orginal in_features
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    state_dict = torch.load('models/watermark_model_v1.pt')

    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    # all_pairs = load_mm(os.path.join(args.json_input_dir, f"{args.image_type}.json"))
    cache_images_ids = []
    cache_preprocessed = []
    input_dir = args.image_input_dir
    allcnt = 0
    with torch.no_grad():
        for file in tqdm(os.listdir(input_dir)):
            try:
                image_tensor = preprocessing(Image.open(os.path.join(input_dir, file))).unsqueeze(0)
            except:
                #all_pairs[file]['clean_score'] = -1
                print('error')
                continue
            cache_images_ids.append(file)
            cache_preprocessed.append(image_tensor)
            if len(cache_images_ids) // args.batch_size == 0:
                print(len(cache_images_ids))
                batch = torch.cat(cache_preprocessed, dim=0)
                pred = model(batch)
                watermark_score = F.softmax(pred, dim=1)[:,1].detach().numpy().tolist()
                print(watermark_score)
                for idx, image_id in enumerate(cache_images_ids):
                    if watermark_score[idx] < 0.3:
                        allcnt -= 1
                cache_images_ids = []
                cache_preprocessed = []
            allcnt += 1
    print(allcnt)




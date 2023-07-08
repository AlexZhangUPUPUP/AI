import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

from wenlan import build_network
from wenlan.utils import getLanMask
from wenlan.utils.config import cfg_from_yaml_file, cfg


parser = argparse.ArgumentParser()
parser.add_argument('--load_checkpoint', type=str, default='/mnt/yzd/checkpoints/brivl-with-roberta-base.pth')
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--img_bsize', type=int, default=8) # adjust the value according to GPU memory 64
parser.add_argument('--txt_bsize', type=int, default=8) # adjust the value according to GPU memory
parser.add_argument('--max_text_len', type=int, default=34) # adjust the value according to the maximum number of Chinese characters in each piece of text
                                                            # if the maximum number of Chinese characters for all texts is N, then this value should be at least N+2
                                                            # this value should not be more than 80
parser.add_argument('--data_root', type=str, default='/mnt/yzd/Wenlan/data/imgs_wudao/') # your path to the folder of images
parser.add_argument('--seed', type=int, default=222)
parser.add_argument('--cfg_file', type=str, default='./wenlan/cfg/eval.yml')
args = parser.parse_args()
cfg_from_yaml_file(args.cfg_file, cfg)
if args.max_text_len < cfg.MODEL.MAX_TEXT_LEN:
    cfg.MODEL.MAX_TEXT_LEN = args.max_text_len

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
torch.manual_seed(args.seed) # cpu
torch.cuda.manual_seed(args.seed) #gpu
np.random.seed(args.seed) #numpy
random.seed(args.seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

##### load the pre-trained model
print('Loading the pre-trained model...')
model = build_network(cfg.MODEL)
model = model.cuda()
model_component = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))
model.learnable.load_state_dict(model_component['learnable'])
img_encoder = model.learnable['imgencoder'].eval()
txt_encoder = model.learnable['textencoder'].eval()
print('Done')

import webdataset as wds
from torchvision import transforms
from transformers import AutoTokenizer
# from PIL import Image
file = "/mnt/datasets/multimodal/wukong/img/00001.tar"#{00000..00003}
dataset = wds.WebDataset(file).decode("rgb").to_tuple("jpg","json")#.shuffle(16)
#print(isinstance(dataset, torch.utils.data.IterableDataset))
import matplotlib.pyplot as plt
i=0
if not os.path.exists("wukong_img"):
    os.mkdir("wukong_img") 
for img,json in dataset:
    # if i<8:
    #     i+=1
    #     continue
    text =  json["caption"]
    #print(type(img))
    text=text.replace("/","-")
    plt.imsave("wukong_img/{}-{}.jpg".format(i,text),img)
    i+=1
    if i==8:
        break

image_size = cfg.MODEL.IMG_SIZE
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                        std=[0.229, 0.224, 0.225])
])
text_transform = AutoTokenizer.from_pretrained(cfg.MODEL.ENCODER)
max_text_len = cfg.MODEL.MAX_TEXT_LEN
def preprocess(sample):
    image, json = sample
    image = transform(image)
    img_box_s = []
    new_size = cfg.MODEL.IMG_SIZE
    box_grid = cfg.MODEL.BOX_GRID
    for i in range(box_grid):
        for j in range(box_grid):
            img_box_s.append(torch.from_numpy(np.array([i * (new_size / box_grid), j * (new_size / box_grid), (i+1) * (new_size / box_grid), (j+1) * (new_size / box_grid)])))
    img_box_s.append(torch.from_numpy(np.array([0, 0, new_size, new_size]).astype(np.float32))) # bbox number:  cfg.MODEL.MAX_IMG_LEN

    valid_len = len(img_box_s)
    img_len = torch.full((1,), valid_len, dtype=torch.long)

    if valid_len < cfg.MODEL.MAX_IMG_LEN:
        for i in range(cfg.MODEL.MAX_IMG_LEN - valid_len):
            img_box_s.append(torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)))

    image_boxs = torch.stack(img_box_s, 0) # <36, box_grid>
    #text:
    text = json["caption"]
    text_info = text_transform(text, padding='max_length', truncation=True,
                                        max_length=max_text_len, return_tensors='pt')
    text = text_info.input_ids.reshape(-1)
    text_len = torch.sum(text_info.attention_mask)
    return image, img_len, image_boxs, text, text_len


dataset = dataset.map(preprocess)

loader = DataLoader(
    dataset,
    batch_size = args.img_bsize,
    shuffle = False,
    num_workers = 0,#8
    pin_memory = True,
    drop_last = False
)
##### extract features
imgFea_all = []
txtFea_all = []
import time
with torch.no_grad():
    for i, batch in enumerate(loader):
        # print(i)
        # if i==0:
        #     continue
        start=time.time()
        images, img_lens, img_boxs = batch[0], batch[1].reshape(-1), batch[2]

        images = images.cuda()
        img_boxs = img_boxs.cuda()

        # get image mask
        imgMask = getLanMask(img_lens, cfg.MODEL.MAX_IMG_LEN)
        imgMask = imgMask.cuda()

        imgFea = img_encoder(images, imgMask, img_boxs)
        imgFea_l2 = F.normalize(imgFea, p=2, dim=-1)

        imgFea_all.append(imgFea_l2)

        #texts:
        texts, text_lens = batch[3], batch[4]
        texts = texts.cuda()
        
        # get language mask
        textMask = getLanMask(text_lens, args.max_text_len)
        textMask = textMask.cuda()

        txtFea = txt_encoder(texts, textMask)
        txtFea_l2 = F.normalize(txtFea, p=2, dim=-1)

        txtFea_all.append(txtFea_l2)
        temp = [''.join(text_transform.convert_ids_to_tokens(text)) for text in texts]
        print("inference time:{:.2f}s".format(time.time()-start))
        for t in temp:
            print(t)
        break
    imgFea_all = torch.cat(imgFea_all, 0)
    txtFea_all = torch.cat(txtFea_all, 0)
    ##### compute similarities
    similarity_matrix = torch.mm(imgFea_all, txtFea_all.t())
    print(similarity_matrix.size())
    print(similarity_matrix)




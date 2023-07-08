import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

from wenlan.wenlan_datasets import ImageData, TextData
from wenlan import build_network
from wenlan.utils import getLanMask
from wenlan.utils.config import cfg_from_yaml_file, cfg


parser = argparse.ArgumentParser()
parser.add_argument('--load_checkpoint', type=str, default='/mnt/yzd/checkpoints/brivl-with-roberta-base.pth')
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--img_bsize', type=int, default=7) # adjust the value according to GPU memory
parser.add_argument('--txt_bsize', type=int, default=7) # adjust the value according to GPU memory
parser.add_argument('--max_text_len', type=int, default=32) # adjust the value according to the maximum number of Chinese characters in each piece of text
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

device="cuda"
##### load the pre-trained model
print('Loading the pre-trained model...')
model = build_network(cfg.MODEL)
model = model.to(device)
model_component = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))
model.learnable.load_state_dict(model_component['learnable'])
img_encoder = model.learnable['imgencoder'].eval()
txt_encoder = model.learnable['textencoder'].eval()
print('Done')

##### image data
img_set = ImageData(cfg, args.data_root)
print(img_set.data)
img_loader = DataLoader(
    img_set,
    batch_size = args.img_bsize,
    shuffle = False,
    num_workers = 8,
    pin_memory = True,
    drop_last = False
)

##### extract image features
imgFea_all = []
with torch.no_grad():
    for i, batch in enumerate(img_loader):
        images, img_lens, img_boxs = batch[0], batch[1].reshape(-1), batch[2]
        images = images.cuda()
        img_boxs = img_boxs.cuda()

        # get image mask
        imgMask = getLanMask(img_lens, cfg.MODEL.MAX_IMG_LEN)
        imgMask = imgMask.cuda()

        imgFea = img_encoder(images, imgMask, img_boxs)
        imgFea_l2 = F.normalize(imgFea, p=2, dim=-1)

        imgFea_all.append(imgFea_l2)

        imf = imgFea.unsqueeze(1)
        break
        
    imgFea_all = torch.cat(imgFea_all, 0)

##### text data
txt_set = TextData(cfg,path="/mnt/datasets/multimodal/wudao/wudao_eg/wudao_titles.txt")
print(txt_set.data, len(txt_set.data))

txt_loader = DataLoader(
    txt_set,
    batch_size = args.txt_bsize,
    shuffle = False,
    num_workers = 8,
    pin_memory = True,
    drop_last = False
)

##### extract text features
txtFea_all = []
with torch.no_grad():
    for i, batch in enumerate(txt_loader):
        texts, text_lens = batch[0], batch[1]
        texts = texts.cuda()
        
        # get language mask
        textMask = getLanMask(text_lens, args.max_text_len)
        textMask = textMask.cuda()

        txtFea = txt_encoder(texts, textMask)
        txtFea_l2 = F.normalize(txtFea, p=2, dim=-1)

        txtFea_all.append(txtFea_l2)
        tf = txtFea.unsqueeze(0)
        np.save("/data/temp/tf.npy",tf.cpu().numpy())
        break
    txtFea_all = torch.cat(txtFea_all, 0)

    ##### compute similarities
    similarity_matrix = torch.mm(imgFea_all, txtFea_all.t())
    print(similarity_matrix.size())
    print(similarity_matrix)

    sim = torch.cosine_similarity(imgFea,txtFea).cpu().numpy()
    simMatrix = torch.cosine_similarity(imf,tf,dim=-1).cpu().numpy()
print("cos similarity:",sim) 
print("cos similarity Matrix:")
print(simMatrix)
room=torch.cuda.max_memory_allocated(device)/(1024*1024)#MB
print("cuda memory usage:{:.2f}MB".format(room))


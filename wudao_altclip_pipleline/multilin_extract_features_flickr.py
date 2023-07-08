#env: pytorch
import torch
from datasets.flickr30kCN import Flickr30kCNtxt,Flickr30kCNimg
from torch.utils.data import DataLoader
from tqdm import tqdm
from multilingual_clip import pt_multilingual_clip
import transformers
import json
device = "cuda"
#prepare model: multilingual clip text encoder + clip vision encoder
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
# Load Model & Tokenizer
txt_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
import clip
vis_model, transform = clip.load("ViT-B/32", device=device,download_root="/mnt/clip_models/")


pair_pth = "/mnt/datasets/multimodal/flickr30k-cn/test_texts.jsonl"
img_dir = "/mnt/datasets/multimodal/flickr30k-images"
txtloader = DataLoader(Flickr30kCNtxt(pair_pth,tokenizer),batch_size=128,num_workers=16)
imgloader = DataLoader(Flickr30kCNimg(pair_pth,img_dir,transform),batch_size=128,num_workers=16)
txt_model.eval()
vis_model.eval()
txt_features = []
img_features = []
img_ids = []
with torch.no_grad():
    for txts in tqdm(txtloader):
        txt_fea = txt_model.forward(list(txts), tokenizer)
        txt_fea /= txt_fea.norm(dim=-1, keepdim=True)
        txt_features.extend(txt_fea.tolist())#若太大则每个batch写入文件
    for batch in tqdm(imgloader):
        ids, imgs = batch[0],batch[1]
        img_ids.extend(ids.tolist())
        imgs = imgs.to(device)
        img_fea = vis_model.encode_image(imgs)
        img_fea /= img_fea.norm(dim=-1, keepdim=True)
        img_features.extend(img_fea.tolist())
name = "flickrCN_multi"
print(len(txt_features),len(txt_features[0]),len(img_features),len(txt_features[0]))
with open("features_save/{}_txt_feat.jsonl".format(name),"w") as f:
    for id,txt_feature in enumerate(txt_features):
        f.write("{}\n".format(json.dumps({"text_id":id,"feature":txt_feature})))
with open("features_save/{}_img_feat.jsonl".format(name),"w") as f:
    for id,img_feature in zip(img_ids,img_features):
        f.write("{}\n".format(json.dumps({"image_id":id,"feature":img_feature})))
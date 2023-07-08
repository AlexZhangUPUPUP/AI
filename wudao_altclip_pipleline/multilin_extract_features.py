#env: pytorch
import torch
from datasets.wudao_test import WudaoTest
from torch.utils.data import DataLoader
from tqdm import tqdm
from multilingual_clip import pt_multilingual_clip
import transformers
import numpy as np
import pandas as pd
import json,random
device = "cuda"

#prepare model: multilingual clip text encoder + clip vision encoder
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
# Load Model & Tokenizer
txt_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
import clip
vis_model, transform = clip.load("ViT-B/32", device=device,download_root="/mnt/clip_models/")

#可修改的参数：
name = "wudao_multi_filt"#用于修改保存文件名前缀
filt = True#是否过滤 若过滤则只存储大于相似度阈值的图文特征
similarity_thr=0.26#图文相似度阈值
N=1000#过滤后抽取的个数（保证测试数据量相同）

dataset=WudaoTest("/mnt/datasets/multimodal/wudao/wudao_test/pairs3k.txt",transform,tokenizer)
dataloader = DataLoader(dataset,batch_size=512,num_workers=32)
#txt_model.to(device)
txt_model.eval()
vis_model.eval()
txt_features = []
img_features = []
sims = []
retains = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        txts, imgs = batch
        #txts = txts.to(device)
        imgs = imgs.to(device)
        txt_fea = txt_model.forward(list(txts), tokenizer)
        txt_fea = txt_fea.to(device)
        img_fea = vis_model.encode_image(imgs)
        txt_fea /= txt_fea.norm(dim=-1, keepdim=True)
        img_fea /= img_fea.norm(dim=-1, keepdim=True)
        if filt:
            #过滤相似度低于similarity_thr的
            sim = torch.cosine_similarity(img_fea,txt_fea).cpu().tolist()
            sims.extend(sim)
            retain = [True if s>similarity_thr else False for s in sim]
            retains.extend(retain)
            txt_fea = txt_fea.cpu().numpy()[retain,:]
            img_fea = img_fea.cpu().numpy()[retain,:]

            txt_features.extend(txt_fea.tolist())#若太大则每个batch写入文件
            img_features.extend(img_fea.tolist())
        else:
            txt_features.extend(txt_fea.cpu().tolist())
            img_features.extend(img_fea.cpu().tolist())
        #print(txt_fea.shape,img_fea.shape) (batch_size,512)
ids=[]#保留图文对id 与txt_features, img_features对齐
if filt:
    for i,r in enumerate(retains):
        if r==True:
            ids.append(i)
    print("过滤后图文对数量：")
    print(len(ids),len(txt_features))
    pd.DataFrame(list(zip(retains,sims)),columns=["retain","sim"]).to_csv("features_save/sims.csv")
    print("全部相似度数量",len(sims))
    print("均值%.3f,方差%.3f"%(np.mean(sims),np.std(sims)))
else:
    ids=list(range(len(dataset)))

if len(ids)>N:#抽样
    sample = random.sample(range(len(ids)),1000)#抽样id
    ids = [ids[i] for i in sample]
    txt_features = [txt_features[i] for i in sample]
    img_features = [img_features[i] for i in sample]
# print("throwing away ids:")
# print(set(list(range(len(dataset))))-set(ids))
print("抽样后数量：")
print(len(txt_features),len(img_features))#,len(txt_features[0])
with open("features_save/{}_txt_feat.jsonl".format(name),"w") as f:
    for id,txt_feature in zip(ids,txt_features):
        f.write("{}\n".format(json.dumps({"text_id":id,"feature":txt_feature})))
with open("features_save/{}_img_feat.jsonl".format(name),"w") as f:
    for id,img_feature in zip(ids,img_features):
        f.write("{}\n".format(json.dumps({"image_id":id,"feature":img_feature})))
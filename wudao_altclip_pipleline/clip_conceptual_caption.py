import torch
import clip
from PIL import Image
import pandas as pd
import random
filepth="/mnt/datasets/multimodal/ConceptualCaptions/Train_GCC-training_output.csv"
df = pd.read_csv(filepth,sep="\t")
n=5 #number of sample
idx=random.sample(range(len(df.index)),n)
img_pths = df["filepath"].iloc[idx].tolist()
img_pths = ["/mnt/datasets/multimodal/ConceptualCaptions/"+t for t in img_pths]
texts = df["title"].iloc[idx].tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device,download_root="/mnt/clip_models/")

import time
for i,pth in enumerate(img_pths):
    if i==0:
        t0=time.time()
        images=preprocess(Image.open(pth)).unsqueeze(0).to(device)
        print("transform a image time:",time.time()-t0)
        continue
    else:
        images = torch.cat((images,preprocess(Image.open(pth)).unsqueeze(0).to(device)),0)    

text_inputs = clip.tokenize(texts,truncate=True).to(device)    
start=time.time()
with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text_inputs)
    imf = image_features.unsqueeze(1)
    tf = text_features.unsqueeze(0)
    sim = torch.cosine_similarity(image_features,text_features).cpu().numpy()
    simMatrix = torch.cosine_similarity(imf,tf,dim=-1).cpu().numpy()
end=time.time()
print("similarity:",sim,"model inference times:{:.3f}s".format(end-start)) 
print("similarity Matrix:")
print(simMatrix)
room=torch.cuda.max_memory_allocated(device)/(1024*1024)#MB
print("cuda memory usage:{:.2f}MB".format(room))
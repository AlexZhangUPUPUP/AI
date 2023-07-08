import torch
from PIL import Image
import numpy as np
dir = "/mnt/datasets/multimodal/wudao/wudao_eg/"
img_pths = [dir+"imgs/00"+str(n)+".jpg" for n in range(1,8)]
with open(dir+"titles_en.txt","r") as f:
    texts = f.readlines()
    texts = [t.strip() for t in texts]
for t in texts:
    print(t)
device = "cuda" if torch.cuda.is_available() else "cpu"
import clip
# model, preprocess = clip.load("ViT-B/32", device=device,download_root="/mnt/clip_models/")
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
model.to(device)
#print(model)
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
    #tf = np.load("/data/temp/txt_embed_en.npy")#for other txt encoder 
    tf = torch.tensor(tf,device=device)
    sim = torch.cosine_similarity(image_features,text_features).cpu().numpy()
    simMatrix = torch.cosine_similarity(imf,tf,dim=-1).cpu().tolist()
end=time.time()
print("similarity:",sim,"model inference times:{:.3f}s".format(end-start)) 
print("similarity Matrix:")
for r in simMatrix:
    for n in r:
        print('%.3f '%n,end='')
    print()
#print(simMatrix)
room=torch.cuda.max_memory_allocated(device)/(1024*1024)#MB
print("cuda memory usage:{:.2f}MB".format(room))
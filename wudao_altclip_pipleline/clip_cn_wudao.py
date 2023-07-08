import json
import torch
from clip_cn.model import CLIP
vision_model_config_file="./clip_cn/model_configs/ViT-B-16.json"
text_model_config_file="./clip_cn/model_configs/RoBERTa-wwm-ext-base-chinese.json"
chkpt="/mnt/clip_models/clip_cn_vit-b-16.pt"
with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
    model_info = json.load(fv)
    for k, v in json.load(ft).items():
        model_info[k] = v

model = CLIP(**model_info)
checkpoint = torch.load(chkpt, map_location='cpu')
start_epoch = checkpoint["epoch"]
sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
model.load_state_dict(sd)

print("loaded checkpoint")

from torch.utils.data import Dataset, DataLoader
from clip_cn.clip import tokenize
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
batch_size=7
class Dataset(Dataset):#参数加transform tokenizer 分模型而不同
    def __init__(self,filepth,imgdir,max_txt_length=34,resolution=224):
        self.texts=[]
        self.imgpths=[imgdir+"00{}.jpg".format(i) for i in range(1,batch_size+1)]
        print(self.imgpths)
        with open(filepth,"r") as f:
            for line in f:
                line=line.strip()
                self.texts.append(line)
        self.max_txt_length = max_txt_length
        self.transform = self._build_transform(resolution)
    def _build_transform(self, resolution):
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return Compose([
                Resize((resolution, resolution), interpolation=Image.BICUBIC),
                self._convert_to_rgb,
                ToTensor(),
                normalize,
            ])
    def _convert_to_rgb(self,image):
        return image.convert('RGB')
    def _preprocess_text(self,text):
        # adapt the text to Chinese BERT vocab
        text = text.lower().replace("“", "\"").replace("”", "\"")
        return text
    def __len__(self):
        return len(self.texts)
    def __getitem__(self,idx):
        text=self.texts[idx]
        #print(text)
        text = tokenize([self._preprocess_text(str(text))], context_length=self.max_txt_length)[0]
        img=Image.open(self.imgpths[idx])
        img = self.transform(img)
        return text,img

dataset = Dataset(
        "/mnt/datasets/multimodal/wudao/wudao_eg/wudao_titles.txt",
        '/mnt/yzd/Wenlan/data/imgs_wudao/')
print(1)
loader=DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
device="cuda"
model.eval().to(device)
import time
start=time.time()
for i,batch in enumerate(loader):
    with torch.no_grad():
        #print(len(batch),batch[0].shape,batch[1].shape)
        #2 torch.Size([7, 34]) torch.Size([7, 3, 224, 224]) batch:list
        txts, imgs = batch
        txts = txts.cuda()
        imgs = imgs.cuda()
        txt_features = model(None,txts)
        img_features = model(imgs,None)
        imf = img_features.unsqueeze(1)
        tf = txt_features.unsqueeze(0)
        sim = torch.cosine_similarity(img_features,txt_features).cpu().numpy()
        simMatrix = torch.cosine_similarity(imf,tf,dim=-1).cpu().numpy()
end=time.time()
print("similarity:",sim,"model inference times:{:.3f}s".format(end-start)) 
print("similarity Matrix:")
for r in simMatrix:
    for n in r:
        print('%.3f '%n,end='')
    print()
room=torch.cuda.max_memory_allocated(device)/(1024*1024)#MB
print("cuda memory usage:{:.2f}MB".format(room))

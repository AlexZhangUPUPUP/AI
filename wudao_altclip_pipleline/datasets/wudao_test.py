#从悟道数据集中抽出1k个左右样例
from torch.utils.data import Dataset
from PIL import Image
class WudaoTest(Dataset):
    def __init__(self,filepth,transform,tokenizer,max_txt_len=34):
        self.texts=[]
        self.imgpths=[]
        with open(filepth,"r") as f:
            for line in f.readlines():
                line=line.strip()
                imgpth,txt = line.split('\t')
                self.texts.append(txt)
                self.imgpths.append(imgpth)
        self.max_txt_len = max_txt_len
        self.transform = transform
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self,idx):
        text=self.texts[idx]
        #print(text)
        #text = self.tokenizer(text)
        img=Image.open(self.imgpths[idx])
        img = self.transform(img)
        return text,img
    def get_txt(self,idx):
        return self.texts[idx]
    def get_img_pth(self,idx):
        return self.imgpths[idx]
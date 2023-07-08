from torch.utils.data import Dataset
from PIL import Image
import jsonlines,os

class Flickr30kCNtxt(Dataset):
    def __init__(self,filepth,tokenizer,max_txt_len=34) -> None:
        super().__init__()
        self.texts=[]
        self.imgpths=[]
        with jsonlines.open(filepth,"r") as f:
            self.texts = [line["text"] for line in f]                
        self.max_txt_len = max_txt_len
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self,idx):
        text=self.texts[idx]
        return text
class Flickr30kCNimg(Dataset):
    def __init__(self,filepth,imgdir,transform) -> None:
        super().__init__()
        with jsonlines.open(filepth,"r") as f:
            ids = [line["image_ids"][0] for line in f]
        self.ids = list(set(ids))
        self.imgpths = [os.path.join(imgdir,str(fn)+".jpg") for fn in self.ids]
        self.transform = transform
    def __len__(self):
        return len(self.imgpths)
    def __getitem__(self,idx):
        img=Image.open(self.imgpths[idx])
        img = self.transform(img)
        return self.ids[idx],img

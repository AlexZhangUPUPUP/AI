from multilingual_clip import pt_multilingual_clip
import transformers
import numpy as np
txt_pth = "/mnt/datasets/multimodal/wudao/wudao_eg/titles_en.txt"
#中文："/mnt/datasets/multimodal/wudao/wudao_eg/wudao_titles.txt"
with open(txt_pth,"r") as f:
    texts = f.readlines()
    texts = [t.strip() for t in texts]
    print(texts)

model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'

# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

embeddings = model.forward(texts, tokenizer)
print(embeddings.shape)
np.save("/data/temp/txt_embed_en.npy",embeddings.detach().numpy())
#then run clip_wudao_en.py to get img features 
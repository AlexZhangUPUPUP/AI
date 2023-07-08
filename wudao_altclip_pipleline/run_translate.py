import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import ipdb
import json
import argparse
import os
import time
import random
p = argparse.ArgumentParser()

p.add_argument('--workers', type = int, default=32)
p.add_argument('--image_input_dir', type = str, help='Directory to input image')
p.add_argument('--image_type', type = str, help='image_type')
p.add_argument('--json_input_dir', type = str, help='Directory to input image')
p.add_argument('--output_dir', type = str, help='Directory to save result to')
p.add_argument('--batch_size', type = int, default = 128, help='Directory to save result to')
p.add_argument('--model_path', type = str, help='Directory to model')
p.add_argument('--shard_id', type = int, default=0, help='shard id')
p.add_argument('--shard_size', type = int, default=0, help='shard id')
args = p.parse_args()

# from transformers import MarianTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wudao_art_path = os.path.join(f'{args.json_input_dir}',f'{args.image_type}_safety_aesthetics_watermark_caption_{args.shard_id}.json')

class CaptionDataset(Dataset):
    def __init__(self, tokenizer, path = wudao_art_path):
        with open(path, 'r') as f:
            self.data = list(json.load(f))
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokens = self.tokenizer(item['caption_blip'], return_tensors="pt")
        return (tokens, item)


def custom_collate_fn(data):
    """
    Data collator with padding.
    """
    tokens = [sample[0]["input_ids"][0] for sample in data]
    attention_masks = [sample[0]["attention_mask"][0] for sample in data]
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
    padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    items = [sample[1] for sample in data]
    batch = {"input_ids": padded_tokens, "attention_mask": attention_masks, 'raw_items': items}
    return batch

tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained(f"{args.model_path}/opus-mt-en-zh")
# if torch.cuda.is_available():
#     model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()
model.eval()

test_data = CaptionDataset(tokenizer)
test_dataloader = DataLoader(
    test_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=32,
    collate_fn=custom_collate_fn,
)

all_pairs = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_dataloader)):
        raw_items = batch.pop('raw_items')
        batch = {k: v.to(device) for k, v in batch.items()}
        output_tokens = model.generate(**batch)
        decoded_tokens = tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)
        for idx, item in enumerate(raw_items):
            item['caption_blip'] = decoded_tokens[idx]
            all_pairs.append(item)

with open(f"{args.image_input_dir}/{args.image_type}_safety_aesthetics_watermark_caption_zh_{args.shard_id}.json", 'w') as f:
    json.dump(all_pairs, f, ensure_ascii=False)
import open_clip, Image
from add_columns import load_mm
import argparse
import os
import multiprocess
import json
import time
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
device = "cuda:0"
model.to(device)

p = argparse.ArgumentParser()
p.add_argument('--workers', type = int, default=32)
p.add_argument('--image_input_dir', type = str, help='Directory to input image')
p.add_argument('--image_type', type = str, help='image_type')
p.add_argument('--json_input_dir', type = str, help='Directory to input image')
p.add_argument('--output_dir', type = str, help='Directory to save result to')

args = p.parse_args()

image=preprocess(Image.open(args.input_path)).unsqueeze(0).to(device)

all_pairs = load_mm(os.path.join(args.json_input_dir, f"{args.image_type}.json"))
pool = multiprocessing.Pool(args.workers)
print(model.config)
time.sleep(10)
image_cnt = 0
image_dict = []

def image_infer(input_path):
    for sub_dir in os.listdir(input_path):
        sub_dir_abs = os.path.join(input_path, sub_dir)
        if os.path.isdir(sub_dir_abs):
            for file in os.listdir(sub_dir_abs):
                image_p = preprocess(Image.open(os.path.join(sub_dir_abs, file))).unsqueeze(0).to(device)
                image_emb = model.encode_image(image_p)
                yield (file, image_emb)

image_embs = np.memmap(os.path.join(output_dir, f"{args.image_type}.memmap"), shape=(600000, model.config['hidden_size']), mode='w', dtype=np.float32)
image_embs = pool.imap(model.encode_image, load_mm(args.input_dir), 32)

for file, image_emb in image_embs:
    image_dict[image_cnt] = file
    image_embs[image_cnt] = image_emb
    image_cnt += 1

image_dict.append(image_cnt)
with open(os.path.join(output_dir, f"{args.image_type}.json"), 'w') as f:
    json.dump(image_dict, f, ensure_ascii=True)

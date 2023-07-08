from add_columns import load_mm
import sys
import json
import os

image_type = sys.argv[1]
input_dir = sys.argv[2]
inter_dir = sys.argv[3]
output_dir = sys.argv[4]
infor_pairs = load_mm(os.path.join(inter_dir, f"{image_type}_watermark_safety_aesthetics_caption_zh.json"))
with open(os.path.join(input_dir, f"{image_type}.json"), 'r') as f:
    all_pairs = json.load(f)

for pair in all_pairs:
    if pair['name'] in infor_pairs:
        pair['clean_score'] = infor_pairs[pair['name']]['clean_score'] if 'clean_score' in infor_pairs[pair['name']] else 0.0
        pair['caption_blip'] = infor_pairs[pair['name']]['caption_blip'] if 'caption_blip' in infor_pairs[pair['name']] else ""
        pair['aesthetics_score'] = infor_pairs[pair['name']]['aesthetics_score'] if 'aesthetics_score' in infor_pairs[pair['name']] else -100.0
        pair['safety_score'] = infor_pairs[pair['name']]['safety_score'] if 'safety_score' in infor_pairs[pair['name']] else -100.0
    else:
        pair['clean_score'] = 0.0
        pair['clean_score'] = ""
        pair['aesthetics_score'] = -100.0
        pair['safety_score'] = -100.0

with open(f"{output_dir}/{image_type}.json", 'w') as f:
    json.dump(all_pairs, f, ensure_ascii=False)

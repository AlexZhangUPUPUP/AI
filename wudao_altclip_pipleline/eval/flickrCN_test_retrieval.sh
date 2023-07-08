name=flickrCN_multi
python -u eval/make_topk_predictions.py \
    --image-feats="features_save/${name}_img_feat.jsonl" \
    --text-feats="features_save/${name}_txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="retrieval_result/${name}_predictions.jsonl"
python eval/evaluation.py \
        /mnt/datasets/multimodal/flickr30k-cn/test_texts.jsonl \
        retrieval_result/${name}_predictions.jsonl \
        retrieval_result/${name}_output.json
#text retrieval:
python -u eval/make_topk_predictions_tr.py \
    --image-feats="features_save/${name}_img_feat.jsonl" \
    --text-feats="features_save/${name}_txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="retrieval_result/${name}_tr_predictions.jsonl"
python eval/evaluation_tr.py \
        /mnt/datasets/multimodal/flickr30k-cn/test_texts.jsonl \
        retrieval_result/${name}_tr_predictions.jsonl \
        retrieval_result/${name}_output_tr.json

        
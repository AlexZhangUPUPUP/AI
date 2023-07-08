mkdir $4
mkdir $5
bash run_infer.sh $1 $2 $4 $6 $7
bash run_safety.sh $1 $2 $4 $3 $6 $7
bash run_aesthetics.sh $1 $4 $3 $6 $7
bash run_watermark.sh $1 $2 $4 $6 $7
#cd ../BLIP-shard
#bash run_captions.sh $1 $2 $4 $6 $7
#cd ../MMDatasets-main-shard
#bash run_translate.sh $1 $4 $3 $6 $7
#python script_fix.py $1 $2 $4 $5 $6 $7





# 如何调佣脚本：



# cd /share/projset/webbrain-lijijie/webbrain-lijijie-raw/MMDatasets-main-shard-v2
# bash run_all.sh Art /share/projset/baaishare/baai-mmdataset/wudaomm-5m/ /share/projset/baaishare/baai-mrnd/clip_models/ /home/alex/wudao-inter /home/alex/wudao-output 0 10


# 脚本的输入参数
# bash run_all.sh ${image-domain-name} ${root_path}/${source_input_dir} ${model_path} ${inter_output_dir} ${final_output_dir} ${shard_id} ${shard_num}

# image_domain_name: 输入图片域，和域的图片文件夹名/json前缀名统一
# ${root_path}/${source_input_dir}: 输入图片域的文件夹根目录（包含所有图片文件夹和json）
# model_path：缓存模型位置
# inter_output_dir:  中介产出json存放位置 (最终输出也暂存在此)
# final_output_dir：最终json存放位置 (暂时弃用)
# shard_id: 指定跑数据域的第几个分片
# shard_num: 指定每个分片的大小 （wudaomm-40m每个分片是400000）







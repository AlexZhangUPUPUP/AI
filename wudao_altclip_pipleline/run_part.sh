bash run_safety.sh $1
bash run_aesthetics.sh $1
cd ../BLIP
bash run_captions.sh $1
cd ../MMDatasets-main
bash run_translate.sh $1
python script_fix.py $1

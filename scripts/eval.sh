type='lstm'
conf_thresh=0.01
nms_thresh=0.45
top_k=200
set_file_name='val_video_small'
detection='yes'
gpu_id='1'
if [ $type = 'ssd' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/ssd300_VID2017' \
    --model_name ssd300 \
    --literation 290000 \
    --save_folder ../eval \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name $set_file_name \
    --dataset_name 'VID2017' \
    --tssd $type \
    --gpu_id $gpu_id \
    --detection $detection
elif [ $type = 'lstm' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/tssd300_VID2017_b4_s16_SkipShareReduce_TuneVggExtraLocConf' \
    --model_name 'ssd300' \
    --literation 10000 \
    --save_folder '../eval' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name $set_file_name \
    --dataset_name 'VID2017' \
    --tssd $type \
    --gpu_id $gpu_id \
    --detection $detection
fi

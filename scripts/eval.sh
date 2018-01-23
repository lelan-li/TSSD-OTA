type='lstm'
conf_thresh=0.01
nms_thresh=0.45
top_k=200
set_file_name='val_video_small'
detection='yes'
gpu_id='3'
if [ $type = 'ssd' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/ssd300_VIDDET' \
    --model_name ssd300 \
    --literation 160000 \
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
    --model_dir '../weights/tssd300_VID2017_b4s16_DSkipSi4EpisBack_FixVggExtraLocConf160000' \
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
elif [ $type = 'edlstm' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/tssd300_VID2017_b4s8_ContiED4StepBack_FixVggExtraLocConf160000' \
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
#tssd300_VID2017_b4s8_ContiSi4StepBack_FixVggExtraLocConf160000
#tssd300_VID2017_b4s8_ContiSi4StepBack_FixVggExtraLocConf160000
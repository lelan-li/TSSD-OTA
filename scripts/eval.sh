conf_thresh=0.01
nms_thresh=0.45
top_k=200
type='tblstm'
set_file_name='val'
detection='yes'
gpu_id='1'
attention='yes'
dataset_name='VID2017'
if [ $type = 'ssd' ]
then
    python ../eval.py \
    --model_dir '../weights040/ssd300_UW' \
    --model_name ssd300 \
    --literation 80000 \
    --save_folder ../eval \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name $set_file_name \
    --dataset_name $dataset_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --detection $detection \
    --attention $attention
elif [ $type = 'tblstm' ]
then
    python ../eval.py \
    --model_dir '../weights040/VID/tssd300_VID2017_SAL_812' \
    --model_name 'ssd300' \
    --literation 10000 \
    --save_folder '../eval' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name $set_file_name \
    --dataset_name $dataset_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --detection $detection \
    --attention 'yes' \
    --tub 0 \
    --tub_thresh 0.95 \
    --tub_generate_score 0.5
elif [ $type = 'gru' ]
then
    python ../eval.py \
    --model_dir '../weights/tssd300_VID2017_b8s8_DRSkipAttGru_Drop2Clip5_FixVggExtraPreLocConf' \
    --model_name 'ssd300' \
    --literation 25000 \
    --save_folder '../eval' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name $set_file_name \
    --dataset_name 'VID2017' \
    --tssd $type \
    --gpu_id $gpu_id \
    --detection $detection \
    --attention $attention
fi

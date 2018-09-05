type='tblstm'
conf_thresh=0.01
nms_thresh=0.45
top_k=200
set_file_name='val'
detection='yes'
gpu_id='3'
attention='yes'
if [ $type = 'ssd' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/ssd300res101_VIDDET' \
    --model_name ssd300 \
    --literation 170000 \
    --backbone 'ResNet101' \
    --save_folder ../eval \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name $set_file_name \
    --dataset_name 'VID2017' \
    --tssd 'ssd' \
    --gpu_id $gpu_id \
    --detection $detection \
    --attention $attention
elif [ $type = 'tblstm' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/tssd300res101attrnn_VID2017' \
    --model_name 'ssd300' \
    --literation 5000 \
    --backbone 'ResNet101' \
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
    --attention $attention \
    --refine 'no' \
    --tub 0 \
    --tub_thresh 0.95 \
    --tub_generate_score 0.5
elif [ $type = 'gru' ]
then
    pythonc3 ../eval.py \
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

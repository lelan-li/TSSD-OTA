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
    --model_dir '../weights/attssd300_VIDDET_512' \
    --model_name ssd300 \
    --literation 10000 \
    --save_folder ../eval \
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
elif [ $type = 'lstm' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/tssd300_VID2017_b4s16_DSkipBoth6EpisBack_DropInOut2Clip5_FixVggExtraLocConf160000' \
    --model_name 'ssd300' \
    --literation 50000 \
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
elif [ $type = 'tblstm' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/tssd300_VID2017_b4s16_RSkipAttTBLstm_Drop2Clip5_FixVggExtraLocConf' \
    --model_name 'ssd300' \
    --literation 35000 \
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
elif [ $type = 'tbedlstm' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/tssd300_VID2017_b8s8_RSkipTBDoLstm_Drop2Clip5_FixVggExtraPreLocConf' \
    --model_name 'ssd300' \
    --literation 45000 \
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
elif [ $type = 'gru' ]
then
    pythonc3 ../eval.py \
    --model_dir '../weights/tssd300_VID2017_b4s16_DSkipGru6_RMSPw_DropReUp2Clip5_FixVggExtraLocConf160000' \
    --model_name 'ssd300' \
    --literation 20000 \
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

conf_thresh=0.01
nms_thresh=0.45
top_k=200
type='ssd_resnet'
if [ $type = 'ssd_refine' ]
then
    python ../eval.py \
    --model_dir '../weights040/UW/ssd320RefineFalseDrop_UWb32' \
    --model_name ssd \
    --ssd_dim 320 \
    --literation 20000 \
    --save_folder ../eval \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --backbone 'RefineDet_VGG' \
    --refine 'no' \
    --pm 0.0 \
    --set_file_name 'test' \
    --dataset_name 'UW' \
    --tssd 'ssd' \
    --gpu_id '0' \
    --detection 'yes' \
    --attention 'no'
elif [ $type = 'ssd_resnet' ]
then
    python ../eval.py \
    --model_dir '../weights040/UW/ssd512res50_UW' \
    --model_name ssd \
    --ssd_dim 512 \
    --literation 80000 \
    --save_folder ../eval \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --backbone 'ResNet50' \
    --refine 'no' \
    --pm 0.0 \
    --set_file_name 'test' \
    --dataset_name 'UW' \
    --tssd 'ssd' \
    --gpu_id '0' \
    --detection 'yes' \
    --cuda 'yes' \
    --attention 'no'
elif [ $type = 'tblstm_vid' ]
then
    python ../eval.py \
    --model_dir '../weights040/VID/tssd300_VID2017_SALd15_816' \
    --literation 25000 \
    --save_folder '../eval' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name 'val' \
    --dataset_name 'VID2017' \
    --tssd 'tblstm' \
    --gpu_id '1' \
    --detection 'yes' \
    --attention 'yes' \
    --tub 0 \
    --tub_thresh 0.95 \
    --tub_generate_score 0.5
elif [ $type = 'tblstm_uw' ]
then
    python ../eval.py \
    --model_dir '../weights040/UW/tssd300_UW_SAL_816' \
    --literation 6000 \
    --save_folder '../eval' \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --set_file_name 'test' \
    --dataset_name 'UW' \
    --tssd 'tblstm' \
    --gpu_id '3' \
    --detection 'yes' \
    --attention 'yes' \
    --tub 0 \
    --tub_thresh 0.95 \
    --tub_generate_score 0.5
fi

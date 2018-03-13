type='tblstm'
video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00007010.mp4'
#video_name='/home/sean/data/MOT/MOT17Det/snippets/Venice-2.mp4'
conf_thresh=0.5
nms_thresh=0.45
top_k=7
gpu_id='3'
attention='yes'
if [ $type = 'ssd' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/ssd300_VIDDET_512' \
    --model_name ssd300 \
    --literation 5000 \
    --confidence_threshold $conf_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --attention 'no'
#    --save_dir '../demo/res/whale36000'
elif [ $type = 'tblstm' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/tssd300_VID2017_b8s8_RContiAttTBLstmAsso5_baseDrop2Clip5_FixVggExtraPreLocConf20000' \
    --model_name ssd300 \
    --literation 5000 \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --attention 'yes' \
    --refine 'no'
#    --save_dir '../demo/res/whale36000'
elif [ $type = 'gru' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/tssd300_VID2017_b8s8_DRSkipAttGru_Drop2Clip5_FixVggExtraPreLocConf' \
    --model_name ssd300 \
    --literation 20000 \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --attention 'yes' \
    --refine 'no' \
    --save_dir '../demo/res/car11005_AttGru'
fi

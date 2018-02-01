type='tblstm'
video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00121000.mp4'
conf_thresh=0.5
nms_thresh=0.45
top_k=5
gpu_id='1'
atention='yes'
if [ $type = 'ssd' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/attssd300_VIDDET_512_atthalf' \
    --model_name ssd300 \
    --literation 5000 \
    --confidence_threshold $conf_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --attention $atention
elif [ $type = 'lstm' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/tssd300_VID2017_b4_s16_SkipTanhReduce_FixVggExtraPreLocConf50000' \
    --model_name ssd300 \
    --literation 30000 \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id
elif [ $type = 'outlstm' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/tssd300_VID2017_b8s16_RSkipOutLstm_RMSPw_Clip5_FixVggExtraLocConf10000' \
    --model_name ssd300 \
    --literation 5000 \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id
elif [ $type = 'tblstm' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/tssd300_VID2017_b8s8_RSkipAttTBLstm_RMSPw_Clip5_FixVggExtraLocConf5000' \
    --model_name ssd300 \
    --literation 10000 \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --attention $atention
fi


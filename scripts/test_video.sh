type='ssd'
video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00131000.mp4'
conf_thresh=0.5
nms_thresh=0.45
top_k=5
gpu_id='1'
if [ $type = 'ssd' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/ssd300_VID2017' \
    --model_name ssd300 \
    --literation 290000 \
    --confidence_threshold $conf_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type
    --gpu_id $gpu_id
elif [ $type = 'lstm' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/tssd300_VID2017_b4_s16_SkipShare_TuneVggExtraLocConf' \
    --model_name ssd300 \
    --literation 15000 \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'seqVID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id
fi


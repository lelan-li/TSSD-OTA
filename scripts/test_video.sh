type='ssd'
#video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00131000.mp4'
video_name='/home/sean/data/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00061000.mp4'

conf_thresh=0.5
nms_thresh=0.45
top_k=10
gpu_id='0'
attention='yes'
if [ $type = 'ssd' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/attssd300_VIDDET_512' \
    --model_name ssd300 \
    --literation 5000 \
    --confidence_threshold $conf_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --attention $attention \
    --save_dir '../demo/res/bird61000'
elif [ $type = 'tblstm' ]
then
    pythonc3 ../test_video.py \
    --model_dir '../weights/tssd300_VID2017_b8s8_RSkipAttTBLstm_RMSPw_Clip5_FixVggExtraLocConf5000' \
    --model_name ssd300 \
    --literation 30000 \
    --confidence_threshold $conf_thresh \
    --nms_threshold $nms_thresh \
    --top_k $top_k \
    --ssd_dim 300 \
    --dataset_name 'VID2017' \
    --video_name $video_name \
    --tssd $type \
    --gpu_id $gpu_id \
    --attention 'yes' \
    --save_dir '../demo/res/airplane7010'
fi


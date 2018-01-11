type='share_conv_lstm'
if [ $type = 'ssd' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder ../weights/ssd300_VID2017_mean128/ \
    --step_list 200000 300000 \
    --batch_size 32 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'VID2017' \
    --augm_type 'ssd' \
    --tssd 'ssd'
elif [ $type = 'conf_conv_lstm' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b2_s16_conf_preVggExtra_bicycle/' \
    --step_list 10000 14000 \
    --batch_size 2 \
    --seq_len 32 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'seqssd' \
    --set_file_name 'train_video_4' \
    --tssd 'conf_conv_lstm' \
    --resume_from_ssd '../weights/ssd300_VID2017/ssd300_VID2017_290000.pth' \
    --freeze 'yes'
elif [ $type = 'share_conv_lstm' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b2_s32_SkipShare_preVggExtraLocConf_bycicle/' \
    --step_list 10000 14000 \
    --batch_size 2 \
    --seq_len 32 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'seqssd' \
    --set_file_name 'train_video_4' \
    --tssd 'share_conv_lstm' \
    --resume_from_ssd '../weights/ssd300_VID2017/ssd300_VID2017_290000.pth' \
    --freeze 'no'
elif [ $type = 'both_conv_lstm' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder ../weights/tssd300_VID2017_b2_s16_both_preVggExtra/ \
    --step_list 200000 30000 \
    --batch_size 2 \
    --seq_len 16 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'seqssd' \
    --tssd 'both_conv_lstm' \
    --resume_from_ssd '../weights/ssd300_VID2017/ssd300_VID2017_290000.pth'
fi

# train_video_remove_no_object
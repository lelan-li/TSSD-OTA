type='lstm'
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
elif [ $type = 'lstm' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b4_s16_SkipShare_preVggExtraLocConf/' \
    --step_list 100000 140000 \
    --batch_size 4 \
    --seq_len 16 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'seqVID2017' \
    --augm_type 'seqssd' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'lstm' \
    --resume_from_ssd '../weights/ssd300_VID2017/ssd300_VID2017_290000.pth' \
    --freeze 'no'
fi

# train_video_remove_no_object
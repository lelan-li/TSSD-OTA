type='lstm'
if [ $type = 'ssd' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/ssd300_VIDDET/' \
    --step_list 150000 200000 250000 \
    --batch_size 64 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'VIDDET' \
    --set_file_name 'train_VID_DET' \
    --augm_type 'ssd' \
    --resume_from_ssd 'ssd' \
    --tssd 'ssd'
elif [ $type = 'lstm' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b4_s16_SkipTanhReduce_FixVggExtraPreLocConf50000/' \
    --step_list 50000 80000 \
    --batch_size 4 \
    --seq_len 16 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'lstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_50000.pth' \
    --freeze 'yes'
fi

# train_video_remove_no_object
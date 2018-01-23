type='lstm'
if [ $type = 'ssd' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/ssd300_VIDDET_test/' \
    --step_list 150000 200000 250000 \
    --batch_size 64 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
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
    --save_folder '../weights/tssd300_VID2017_b4s16_DSkipBoth6EpisBack_DropClip_FixVggExtraLocConf160000/' \
    --step_list 50000 80000 100000 \
    --batch_size 4 \
    --seq_len 16 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'lstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_160000.pth' \
    --freeze 'yes'
elif [ $type = 'edlstm' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b4s8_ContiED4StepBack_FixVggExtraLocConf160000/' \
    --step_list 80000 100000 \
    --batch_size 4 \
    --seq_len 8 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'edlstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_160000.pth' \
    --freeze 'yes' \
    --resume '../weights/tssd300_VID2017_b4_s8_ContiEDStepBack_FixVggExtraLocConf160000/ssd300_seqVID2017_10000.pth' \
    --start_iter 10000
fi

# train_video_remove_no_object
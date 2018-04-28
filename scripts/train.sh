type='tblstm'
if [ $type = 'ssd' ]
then
    python ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom 'yes' \
    --send_images_to_visdom 'no' \
    --save_folder '../weights040/ssd300_UW' \
    --step_list 40000 60000 80000 \
    --batch_size 64 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'UW' \
    --set_file_name 'train' \
    --augm_type 'ssd' \
    --num_workers 8 \
    --tssd 'ssd' \
    --attention 'no' \
    --association 'no' \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --freeze 0 \
    --bn 'no' \
    --basenet 'vgg16_reducedfc_512.pth'
#    --resume '../weights/ssd300_MOT1517_bn/ssd300_MOT15_50000.pth' \
#    --start_iter 55000
#    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
elif [ $type = 'tblstm' ]
then
    python ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom yes \
    --send_images_to_visdom no \
    --save_folder '../weights040/tssd300_MOT15_SAL432' \
    --step_list 4000 6000 \
    --batch_size 4 \
    --seq_len 32 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqMOT15' \
    --augm_type 'base' \
    --num_workers 4 \
    --set_file_name 'train_video_remove_no_object' \
    --skip 'yes' \
    --tssd 'tblstm' \
    --freeze 1 \
    --attention 'yes' \
    --association 'no' \
    --asso_top_k 75 \
    --asso_conf 0.9 \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --resume '../weights040/tssd300_MOT15_SAL432/ssd300_seqMOT15_3000.pth' \
    --start_iter 3000
#    --resume '../weights/tssd300_VID2017_b8s8_RSkipAttTBLstm_baseAugmDrop2Clip5d15k_FixVggExtraPreLocConf/ssd300_seqVID2017_20000.pth'
#    --resume_from_ssd '../weights/ssd300_MOT1517/ssd300_MOT15_30000.pth'
elif [ $type = 'gru' ]
then
    python ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom true \
    --save_folder '../weights/tssd300_VID2017_b8s8_DRSkipAttGru_Drop2Clip5_FixVggExtraPreLocConf/' \
    --step_list 15000 30010 \
    --batch_size 2 \
    --seq_len 4 \
    --ssd_dim 300 \
    --gpu_ids '3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'gru' \
    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
    --freeze '1' \
    --attention 'yes'
fi

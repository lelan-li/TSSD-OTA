type='tblstm'
if [ $type = 'ssd' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'yes' \
    --send_images_to_visdom 'no' \
    --save_folder '../weights/ssd300_MOT1517_bn/' \
    --step_list 40000 60000 80000 \
    --batch_size 64 \
    --ssd_dim 300 \
    --gpu_ids '0,1,2,3' \
    --dataset_name 'MOT15' \
    --set_file_name 'train' \
    --augm_type 'ssd' \
    --num_workers 8 \
    --tssd 'ssd' \
    --attention 'no' \
    --association 'no' \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --freeze 0 \
    --bn yes \
    --basenet 'vgg16allbn_reducedfc_512.pth' \
    --resume '../weights/ssd300_MOT1517_bn/ssd300_MOT15_50000.pth' \
    --start_iter 55000
#    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
elif [ $type = 'tblstm' ]
then
    pythonc3 ../train.py \
    --lr 0.00001 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom yes \
    --send_images_to_visdom no \
    --save_folder '../weights/resume6445_to30_50' \
    --step_list 3000 5000 \
    --save_interval 1000 \
    --batch_size 8 \
    --seq_len 8 \
    --ssd_dim 300 \
    --gpu_ids '0,1,2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --num_workers 1 \
    --set_file_name 'train_video_remove_no_object' \
    --skip 'no' \
    --tssd 'tblstm' \
    --freeze 1 \
    --attention 'yes' \
    --refine 'no' \
    --association 'yes' \
    --asso_top_k 75 \
    --asso_conf 0.1 \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
    --resume '../weights/tssd300_VID2017_b8s8_RSkipAttTBLstm_baseAugmDrop2Clip5d15k_FixVggExtraPreLocConf/ssd300_seqVID2017_20000.pth'
#    --start_iter 15000
#    --resume '../weights/tssd300_MOT15_SAL416/ssd300_seqMOT15_2000.pth'
elif [ $type = 'gru' ]
then
    pythonc3 ../train.py \
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

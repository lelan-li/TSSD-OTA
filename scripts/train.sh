type='ssd_uw'
if [ $type = 'ssd_mot' ]
then
    python ../train.py \
    --lr 0.0001 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'yes' \
    --send_images_to_visdom 'no' \
    --save_folder '../weights040/MOT/ssd300_MOT1517' \
    --step_list 15000 25000 30000 \
    --save_interval 5000 \
    --batch_size 64 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'MOT15' \
    --set_file_name 'train' \
    --augm_type 'ssd' \
    --num_workers 8 \
    --tssd 'ssd' \
    --attention 'no' \
    --association 'no' \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --freeze 0 \
    --bn 'no' \
    --basenet 'vgg16_reducedfc_512.pth' \
    --resume '../weights040/MOT/ssd300_MOT1517/ssd300_MOT15_15000.pth' \
    --start_iter 15000
elif [ $type = 'ssd_uw' ]
then
    python ../train.py \
    --lr 0.0001 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom 'yes' \
    --send_images_to_visdom 'no' \
    --save_folder '../weights040/UW/ssd300res50_UW' \
    --model_name ssd \
    --ssd_dim 300 \
    --step_list 30000 40000 50000 \
    --save_interval 10000 \
    --batch_size 64 \
    --gpu_ids '2,3' \
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
    --backbone 'ResNet50' \
    --basenet 'resnet50_reducefc.pth'
#    --resume '../weights040/UW/ssd512_UW/ssd512_UW_10000.pth' \
#    --start_iter 10000
elif [ $type = 'tblstm_vid' ]
then
    python -m torch.utils.bottleneck ../train.py \
    --lr 0.0001 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom yes \
    --send_images_to_visdom no \
    --save_folder '../weights040/VID/tssd300_VID2017_SALd15_812' \
    --step_list 15000 15050 \
    --save_interval 5000 \
    --batch_size 8 \
    --seq_len 12 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --num_workers 1 \
    --set_file_name 'train_video_remove_no_object' \
    --skip 'yes' \
    --tssd 'tblstm' \
    --freeze 1 \
    --attention 'yes' \
    --association 'no' \
    --asso_top_k 75 \
    --asso_conf 0.1 \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
    --resume '../weights040/VID/tssd300_VID2017_SAL_812/ssd300_seqVID2017_15000.pth' \
    --start_iter 15000
elif [ $type = 'tblstm_mot' ]
then
    python ../train.py \
    --lr 0.0001 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom yes \
    --send_images_to_visdom no \
    --save_folder '../weights040/MOT/tssd300_MOT15_SAL_420' \
    --step_list 3000 4000 \
    --save_interval 1000 \
    --batch_size 4 \
    --seq_len 20 \
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
    --asso_top_k 100 \
    --asso_conf 0.1 \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --resume_from_ssd '../weights/ssd300_MOT1517/ssd300_MOT15_30000.pth'
#    --resume '../weights/tssd300_MOT15_SAL222/ssd300_seqMOT15_4000.pth'
elif [ $type = 'tblstm_uw' ]
then
    python ../train.py \
    --lr 0.0001 \
    --gamma 0.1 \
    --momentum 0.9 \
    --visdom yes \
    --send_images_to_visdom yes \
    --save_folder '../weights040/UW/tssd300_UW_SAL_816' \
    --step_list 4000 6000 \
    --save_interval 1000 \
    --batch_size 8 \
    --seq_len 16 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'seqUW' \
    --augm_type 'base' \
    --num_workers 4 \
    --set_file_name 'train_video' \
    --skip 'yes' \
    --tssd 'tblstm' \
    --freeze 1 \
    --attention 'yes' \
    --association 'no' \
    --asso_top_k 75 \
    --asso_conf 0.1 \
    --loss_coe 1.0 1.0 0.5 2.0 \
    --resume_from_ssd '../weights040/UW/ssd300_UW/ssd300_UW_80000.pth'
#    --resume '../weights040/UW/tssd300_seqUW_SAL_816/ssd300_seqUW_2000.pth' \
#    --start_iter 2000
fi

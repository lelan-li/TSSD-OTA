type='ssd'
if [ $type = 'ssd' ]
then
    pythonc3 ../train.py \
    --lr 0.001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/ssd300c512_VIDDET/' \
    --step_list 150000 20000 250000 \
    --batch_size 64 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'VIDDET' \
    --set_file_name 'train_VID_DET' \
    --augm_type 'ssd' \
    --resume_from_ssd 'ssd' \
    --tssd 'ssd'
#    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_160000.pth'
elif [ $type = 'lstm' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b4s16_DSkipLstm6_DropInOut2Clip3_FixVggExtraTuneLocConfRnn10000/' \
    --step_list 15000 30000 \
    --batch_size 4 \
    --seq_len 16 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'lstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_160000.pth' \
    --freeze 'yes' \
    --resume '../weights/tssd300_VID2017_b4s16_DSkipBoth6EpisBack_DropInOut2Clip5_FixVggExtraLocConf160000/ssd300_seqVID2017_10000_rnn.pth'
elif [ $type = 'tblstm' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b8s8_DSkipTBLstm_RMSPw_DropInOut2Clip5_PreVggExtraLocConf160000/' \
    --step_list 20000 30000 40000 \
    --batch_size 8 \
    --seq_len 8 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'tblstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_160000_512.pth' \
    --resume '../weights/tssd300_VID2017_b8s8_DSkipTBLstm_RMSPw_DropInOut2Clip5_FixVggExtraPreLocConf160000/ssd300_seqVID2017_5000.pth' \
    --freeze 'no'
elif [ $type = 'tbedlstm' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b8s8_DSkipTBDoLstm_RMSPw_DropInOut2Clip5_FixVggExtraPreLocConf160000/' \
    --step_list 20000 30000 40000 \
    --batch_size 8 \
    --seq_len 8 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'tbedlstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_160000_512.pth' \
    --freeze 'yes'
elif [ $type = 'gru' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b4s16_DSkipGru6_RMSPw_DropReUp2Clip5_FixVggExtraLocConf160000_test/' \
    --step_list 50000 80000 100000 \
    --batch_size 8 \
    --seq_len 16 \
    --ssd_dim 300 \
    --gpu_ids '0,1s' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'gru' \
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
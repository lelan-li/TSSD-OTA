type='tblstm'
if [ $type = 'ssd' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom 'yes' \
    --send_images_to_visdom 'yes' \
    --save_folder '../weights/attssd300_VIDDET_512/' \
    --step_list 20000 40000 \
    --batch_size 64 \
    --ssd_dim 300 \
    --gpu_ids '0,1' \
    --dataset_name 'VIDDET' \
    --set_file_name 'train_VID_DET' \
    --augm_type 'ssd' \
    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
    --tssd 'ssd' \
    --attention 'yes' \
    --freeze 1
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
    --augm_type 'ssd' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'lstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET/ssd300_VIDDET_160000.pth' \
    --freeze 2 \
    --resume '../weights/tssd300_VID2017_b4s16_DSkipBoth6EpisBack_DropInOut2Clip5_FixVggExtraLocConf160000/ssd300_seqVID2017_10000_rnn.pth'
elif [ $type = 'outlstm' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom false \
    --save_folder '../weights/tssd300_VID2017_b8s8_RSkipOutReluLstm_RMSPw_Clip5_FixVggExtraLocConf10000/' \
    --step_list 20000 30000 40000 \
    --batch_size 8 \
    --seq_len 8 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'outlstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET_186/ssd300_VIDDET_10000.pth' \
    --freeze 'yes'
elif [ $type = 'tblstm' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom true \
    --save_folder '../weights/tssd300_VID2017_b8s8_RSkipAttTBLstm_Drop2Clip5_FixVggExtraLocConf30000/' \
    --step_list 3000 40000 50000 \
    --batch_size 8 \
    --seq_len 8 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'tblstm' \
    --freeze 2 \
    --attention 'yes' \
    --resume '../weights/tssd300_VID2017_b8s8_RSkipAttTBLstm_Drop2Clip5_FixVggExtraLocConf/ssd300_seqVID2017_30000.pth'
#    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
#    --start_iter 15000

elif [ $type = 'tbedlstm' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom true \
    --save_folder '../weights/tssd300_VID2017_b8s8_RSkipTBDoLstm_Drop2Clip5_FixVggExtraPreLocConf/' \
    --step_list 30000 40000 50000 \
    --batch_size 8 \
    --seq_len 8 \
    --ssd_dim 300 \
    --gpu_ids '2,3' \
    --dataset_name 'seqVID2017' \
    --augm_type 'base' \
    --set_file_name 'train_video_remove_no_object' \
    --tssd 'tbedlstm' \
    --resume_from_ssd '../weights/ssd300_VIDDET_512/ssd300_VIDDET_5000.pth' \
    --freeze '2' \
    --attention 'yes'
elif [ $type = 'gru' ]
then
    pythonc3 ../train.py \
    --lr 0.0001 \
    --momentum 0.9 \
    --visdom true \
    --send_images_to_visdom true \
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
pythonc3 ../train.py \
--lr 0.001 \
--momentum 0.9 \
--visdom true \
--send_images_to_visdom false \
--save_folder ../weights/tssd300_VID2017_b2_s16_both_preVggExtra/ \
--step_list 100000 140000 \
--batch_size 2 \
--seq_len 16 \
--ssd_dim 300 \
--gpu_ids '2,3' \
--dataset_name 'seqVID2017' \
--augm_type 'seqssd' \
--tssd 'both_conv_lstm' \
--resume_from_ssd '../weights/ssd300_VID2017/ssd300_VID2017_290000.pth'
# --start_iter 0 \
#--resume ../weights/ssd300_VID2017_noarg/ssd300_VID2017_190000.pth \
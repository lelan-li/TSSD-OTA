pythonc3 ../eval.py \
--model_dir ../weights/ssd300_VID2017_noarg \
--model_name ssd300 \
--literation 220000 \
--save_folder ../eval \
--confidence_threshold 0.01 \
--top_k 5 \
--ssd_dim 300 \
--set_file_name 'val' \
--dataset_name 'VID2017' \
--detection

# --trained_model ../weights/VOC0712/ssd300_0712_115000.pth \
#--trained_model ../weights/VID2017/ssd300_VID2017_20000.pth \


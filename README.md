# Temporally Identity-Aware SSD with Attentional LSTM

## Project
This repository is initially based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), and based on pytorch v0.40
[arXiv paper](https://arxiv.org/abs/1803.00197) describes this project in detail.


If you use this code for your research, please cite:

**Xingyu Chen, Junzhi Yu, and Zhengxing Wu, "Temporally Identity-Aware SSD with Attentional LSTM", *arXiv:1803.00197*, 2018.**

## Useage
### Dataset
Dowload [VID dataset](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php) and place them at
```
/home/xxx/data/ILSVRC
```
This path is defined in `<TSSD-ROOT>/data/config.py`
## Pre-trained model and training list
You can dowload them at [BaiduYun](https://pan.baidu.com/s/1vDorzGcdEtGa0ZeNmRJTJw)
1. The Pre-trained model is under the condition of `batch_size=8, seq_len=12`. You should put it in the following directory
```
<TSSD-ROOT>/weights040/VID/tssd300_VID2017_SAL_812/ssd300_seqVID2017_10000.pth'
```
2. The training list should be put at data folder, e.g.,
```
/home/xxx/data/ILSVRC/ImageSets/VID/train_video_remove_no_object.txt
```

## Train
Set `type='tblstm_vid'` in `scripts/train.sh`, then
```
cd scripts
./train.sh
```

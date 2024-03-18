# MFIS-net: A Deep Learning Framework for Left Atrial Segmentation
by Jie Gui, Wen Sha, Xiuquan Du
## Introduction
This repository is for our paper 'MFIS-net: A Deep Learning Framework for Left Atrial Segmentation.
## Usage
### Install
Clone the repo:
```shell
git clone https://github.com/shaw58/MFIS-net 
```
### Dataset
We use [the dataset of 2018 Atrial Segmentation Challenge](http://atriaseg2018.cardiacatlas.org/).
### Test
If you want to test MFIS-net on LA.
```shell
cd MFIS-net
python ./code/test.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 8 --gpu 0
```
## Acknowledgements
Our code is origin from ASPP, CBAM. Thanks to these authors for their excellent work.

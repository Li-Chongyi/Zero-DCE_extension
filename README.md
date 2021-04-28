# Zero-DCE++

You can find more details here: https://li-chongyi.github.io/Proj_Zero-DCE++.html. 

You can find the details of our CVPR version: https://li-chongyi.github.io/Proj_Zero-DCE.html. 

✌If you use this code, please cite our paper. Please hit the star at the top-right corner. Thanks!

# Pytorch
Pytorch implementation of Zero-DCE++

## Requirements
1. Python 3.7 
2. Pytorch 1.0.0
3. opencv
4. torchvision 0.2.1
5. cuda 10.0

Zero-DCE++ does not need special configurations. Just basic environment. 

Or you can create a conda environment to run our code like this:
conda create --name zerodce++_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch

### Folder structure
Download the Zero-DCE_code++ first.
The following shows the basic folder structure.
```

├── data
│   ├── test_data 
│   └── train_data 
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # Zero-DEC++ network
├── dataloader.py
├── snapshots_Zero_DCE++
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```
### Test: 

cd Zero-DCE++
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

### Train: 
cd Zero-DCE++

```
python lowlight_train.py 
```

##  License
The code is made available for academic research purpose only. This project is open sourced under MIT license.

## Bibtex

```
@inproceedings{Zero-DCE,
 author = {Li, Chongyi and Guo, Chunle Guo and Loy, Chen Change},
 title = {Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation},
 booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
 pages    = {},
 month = {},
 year = {2021}
 doi={10.1109/TPAMI.2021.3063604}
}
```

(Full paper: https://ieeexplore.ieee.org/document/9369102 or arXiv version: https://arxiv.org/abs/2103.00860)

## Contact
If you have any questions, please contact Chongyi Li at lichongyi25@gmail.com or Chunle Guo at guochunle@tju.edu.cn.

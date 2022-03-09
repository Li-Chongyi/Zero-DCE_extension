# Zero-DCE++

You can find more details here: https://li-chongyi.github.io/Proj_Zero-DCE++.html. 

You can find the details of our CVPR version: https://li-chongyi.github.io/Proj_Zero-DCE.html. 

✌If you use this code, please cite our paper. Please hit the star at the top-right corner. Thanks!

We also provided a MindSpore version of our code: https://pan.baidu.com/s/1kEjKtYYSwvzHCzDh_4niew (passwords: 37ne). 

🌈We released a survey on deep learning-based low-light image enhancement------- Low-Light Image and Video Enhancement Using Deep Learning: A Survey, an online platform, a new dataset. Have fun! https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open. 

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
Download the Zero-DCE++ first.
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
The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International License.

## Bibtex

```
@inproceedings{Zero-DCE++,
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
If you have any questions, please contact Chongyi Li at lichongyi25@gmail.com or Chunle Guo at guochunle@nankai.edu.cn.

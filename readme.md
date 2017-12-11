# Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach

This repository includes code for the network presented in:

> Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, Yichen Wei, 
> **Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach**
> ICCV 2017 ([arXiv:1704.02447](https://arxiv.org/abs/1704.02447))

The code is developed upon [Stacked Hourglass Network](https://github.com/anewell/pose-hg-train).

**\[New\]** Checkout our [PyTorch implementation](https://github.com/xingyizhou/Pytorch-pose-hg-3d).

Contact: [zhouxy13@fudan.edu.cn](mailto:zhouxy13@fudan.edu.cn)

## Requirements
- cudnn
- [Torch7](https://github.com/torch/torch7) with hdf5 and image
- Python with h5py and opencv

## Testing
- Download our pre-trained [model](http://xingyizhou.xyz/hgreg-3d.t7) and move it to `models`.
- Run `th demo.lua /path/to/image`. 

We provide example images in `src/images/`. For testing your own image, it is important that the person should be at the center of the image and most of the body parts should be within the image. 

## Training
- Prepare the training data:
  - Download our pre-processed Human3.6M dataset [here](https://drive.google.com/open?id=0BxjtxDYaOrYPRlJJeDhfUVAzM00). The main part of the data pre-processing code is in `src/Tools/h36mPreprocessing.m`. We Converted Human3.6M dataset to .jpg files with 5x down-sampling.
  - Run `python GetH36M.py` in `src/Tools/` to convert H36M annotations to hdf5 format.
  - Run `python GetMPI-INF-3D.py` in `src/Tools/` to convert 3DHP annotations to hdf5 format. (or set `valid3DHP` in `opt.lua` false if you don't evaluate on this dataset)

- Stage1: Train the 2D hourglass component
```
cd src
th main.lua -expID Stage1 -dataset fusion -task pose-hgreg-3d -netType hgreg-3d -varWeight 0.0 -regWeight 0.0  -nEpochs 60
```
Our results of this stage is provided [here](https://drive.google.com/open?id=0BxjtxDYaOrYPVmJxNndiaHN1OGc). Most of the experiments in the paper are based on this model. 

- Stage2: Train without Geometry loss  (drop LR at 40 epochs)
```
th main.lua -expID Stage2 -dataset fusion -task pose-hgreg-3d -loadModel ../models/HGRegS2M2M2_60.t7 -varWeight 0.0 -regWeight 0.1 -dropLR 40 -nEpochs 50
```
- Stage3: Train with Geometry loss
```
th main.lua -expID Stage3 -dataset fusion -task pose-hgreg-3d -loadModel ../exp/fusion/Stage2/model_50.t7 -varWeight 0.01 -regWeight 0.1 -LR 2.5e-5 -nEpochs 10`
```

## Citation

    @InProceedings{Zhou_2017_ICCV,
    author = {Zhou, Xingyi and Huang, Qixing and Sun, Xiao and Xue, Xiangyang and Wei, Yichen},
    title = {Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }

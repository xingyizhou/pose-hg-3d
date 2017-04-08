# Weakly-supervised Transfer for 3D Human Pose Estimation in the Wild

This repository includes code for the network presented in:

> Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, Yichen Wei, 
> **Weakly-supervised Transfer for 3D Human Pose Estimation in the Wild**
> ([arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx))

The code is developed upon [Stacked Hourglass Network](https://github.com/anewell/pose-hg-train).

Contact: [zhouxy13@fudan.edu.cn](mailto:zhouxy13@fudan.edu.cn)

## Requirements
- cudnn
- [Torch7](https://github.com/torch/torch7) with hdf5 and image
- Python with h5py and opencv

## Testing
- Download our pre-trained [model](http://xingyizhou.xyz/hgreg-3d.t7) and move it to `src`.
- Run `th demo.lua /path/to/image`. 

We provide example images in `src/images/`. For testing your own image, it is important that the person should be at the center of the image and most of the body parts should be within the image. 

## Training
Coming soon. 

# SRDC-CVPR2020
Code release for Unsupervised Domain Adaptation via Structurally Regularized Deep Clustering (CVPR2020-Oral).

The paper is avaliable [here](http://arxiv.org/abs/2003.08607).

# Requirements
- Python 3.6.3
- Pytorch 1.1.0

# Dataset
The structure of the dataset should be like
```
Office31
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ amazon_half
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ amazon_half2
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|  |_ ... (omit 28 classes)
|  |_ trash_can
|     |_ <im-1-name>.jpg
|     |_ ...
|_ ...
```
# Training
## [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view)
Replace paths and domains in run_office31.sh with those in your own system.

# Citation
```
@InProceedings{srdc,   
  title={Unsupervised Domain Adaptation via Structurally Regularized Deep Clustering},   
  author={Hui Tang, Ke Chen, and Kui Jia},   
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},   
  year={2020},
}
```

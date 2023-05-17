# WSCL

This is the official PyTorch implementation of our paper:

> [**Weak-to-Strong Consistency Learning for Semisupervised Image Segmentation**](https://ieeexplore.ieee.org/document/10114409)      
> **Xiaoqiang Lu**, Licheng Jiao, Lingling Li, Fang Liu, Xu Liu, Shuyuan Yang, Zhixi Feng, Puhua Chen    
> *Accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS) 2023*

# Getting Started
## Install
```
conda create -n WSCL python=3.7
pip install -r requirements.txt
```

## Data Preparation

### Pre-trained Model
```
mkdir pretrained
cd pretrained
wget https://download.pytorch.org/models/resnet101-63fe2227.pth
mv resnet101-63fe2227.pth resnet101.pth
cd ..
```
or download via the following link
[ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth)

### Dataset
We have processed the original dataset as mentioned in the paper. You can access the processed dataset directly via the following link.

[DFC22](https://drive.google.com/file/d/1FZoZiWW_19TWdWKUz1-Vn-IlTs0QWtfg/view?usp=sharing) | 
[iSAID](https://drive.google.com/file/d/1yBlHpxuWs_X_02O5xAxTpo1tDWcGj6-x/view?usp=sharing) | 
[MER](https://drive.google.com/file/d/1KN0PSoWlC4BPv5QAZdJKrhpvu9F9vdlm/view?usp=sharing) | 
[MSL](https://drive.google.com/file/d/1XSvmXB5rsZUi98-rWWwQtONfop7HJw4M/view?usp=sharing) | 
[Vaihingen](https://drive.google.com/file/d/11aOEO3-Ov3Wcg8o2nBA5GsGBqqRHbjRh/view?usp=sharing) |
[GID-15](https://drive.google.com/file/d/1CITO8Bxf8eG-6le6mZDshDgTyi9BO2ot/view?usp=sharing) | 

### File Organization

```
├── ./pretrained
    └── resnet101.pth
    
├── [Your Dataset Path]
    ├── images
    └── labels
```

# Results

<div style="text-align: center;">
<table>
    <tr>
        <td>Dataset</td> 
        <td>Partition</td>
        <td>Method</td> 
        <td>mIoU</td>
        <td>Dataset</td> 
        <td>Partition</td>
        <td>Method</td>
        <td>mIoU</td>
   </tr>
    <tr>
        <td rowspan="6">DFC22</td>    
        <td rowspan="3">1-8</td>
        <td >baseline</td>
        <td >26.97</td>
        <td rowspan="6">iSAID</td> 
        <td rowspan="3">100</td>
        <td >baseline</td>
        <td >39.91</td>
    </tr>
    <tr>
        <td >LSST</td>
        <td >30.94</td>
        <td >LSST</td>
        <td >46.94</td>
    </tr>
    <tr>
        <td >Ours</td>
        <td >38.00</td>
        <td >Ours</td>
        <td >59.60</td>
    </tr>
    <tr>
        <td rowspan="3">1-4</td>
        <td >baseline</td>
        <td >32.67</td>
        <td rowspan="3">300</td>
        <td >baseline</td>
        <td >60.47</td>
    </tr>
    <tr>
        <td >LSST</td>
        <td >36.40</td>
        <td >LSST</td>
        <td >63.40</td>
    </tr>
    <tr>
        <td >Ours</td>
        <td >38.92</td>
        <td >Ours</td>
        <td >72.33</td>
    </tr>
    <tr>
        <td rowspan="6">MER</td>    
        <td rowspan="3">1-8</td>
        <td >baseline</td>
        <td >43.63</td>
        <td rowspan="6">MSL</td>    
        <td rowspan="3">1-8</td>
        <td >baseline</td>
        <td >50.17</td>
    </tr>
    <tr>
        <td >LSST</td>
        <td >49.68</td>
        <td >LSST</td>
        <td >54.72</td>
    </tr>
    <tr>
        <td >Ours</td>
        <td >51.88</td>
        <td >Ours</td>
        <td >58.23</td>
    </tr>
    <tr>
        <td rowspan="3">1-4</td>
        <td >baseline</td>
        <td >48.19</td>
        <td rowspan="3">1-4</td>
        <td >baseline</td>
        <td >50.26</td>
    </tr>
    <tr>
        <td >LSST</td>
        <td >51.31</td>
        <td >LSST</td>
        <td >56.22</td>
    </tr>
    <tr>
        <td >Ours</td>
        <td >54.85</td>
        <td >Ours</td>
        <td >59.91</td>
    </tr>
    <tr>
        <td rowspan="6">Vaihingen</td>    
        <td rowspan="3">1-8</td>
        <td >baseline</td>
        <td >53.30</td>
        <td rowspan="6">GID-15</td>    
        <td rowspan="3">1-8</td>
        <td >baseline</td>
        <td >61.86</td>
    </tr>
    <tr>
        <td >LSST</td>
        <td >64.09</td>
        <td >LSST</td>
        <td >66.38</td>
    </tr>
    <tr>
        <td >Ours</td>
        <td >66.65</td>
        <td >Ours</td>
        <td >71.42</td>
    </tr>
    <tr>
        <td rowspan="3">1-4</td>
        <td >baseline</td>
        <td >59.30</td>
        <td rowspan="3">1-4</td>
        <td >baseline</td>
        <td >67.90</td>
    </tr>
    <tr>
        <td >LSST</td>
        <td >65.34</td>
        <td >LSST</td>
        <td >71.28</td>
    </tr>
    <tr>
        <td >Ours</td>
        <td >68.81</td>
        <td >Ours</td>
        <td >74.88</td>
    </tr>

</table>
</div>


# Training and Testing
## Training
Change DATASET, SPLIT, and DATASET_PATH as you want in train.py, then run:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
## Testing
Change WEIGHTS, and DATASET_PATH as you want in test.py, then run:
```
CUDA_VISIBLE_DEVICES=0,1 python test.py
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@ARTICLE{10114409,
  author={Lu, Xiaoqiang and Jiao, Licheng and Li, Lingling and Liu, Fang and Liu, Xu and Yang, Shuyuan and Feng, Zhixi and Chen, Puhua},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Weak-to-Strong Consistency Learning for Semisupervised Image Segmentation}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3272552}} 
```

We have other work on semi-supervised remote sensing image segmentation:

- [[IEEE TGRS 2022] LSST](https://github.com/xiaoqiang-lu/LSST) 

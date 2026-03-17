# **DINO-JRS: Joint Registration and Segmentation Meets Vision Foundation Model**


当前很多联合分割和配准的方法都是使用传统的深度卷积神经网络，而随着最近几年大模型和视觉基座模型的出现，越来越多的方法开始探索利用大模型和视觉大模型为自己的领域进行赋能。而最近Meta退出的DINO2和DINO3在提取通用的特征上展现出很强大的效果，已经有不少工作开始研究DINO3在下游任务的适配，但是还没有研究尝试将DINO3强大的视觉特征提取和联合分割和配准这个领域结合起来。因此，我们针对这个研究空缺进行了一个尝试，我们将DINO提取的强大的全局特征通过自己设计的一个3D感知的适配器模块和CNN提取的局部特征进行相互融合，以此获取更强大的领域特征；并且我们注意到还很少人研究配准和分割两个任务的特征相互增强，因此我们也设计了一个模块对两个任务的特征进行了交叉增强。

## Overview
<p align="center">
  <img src="assets/gram.png" alt="MedDINOv3 Framework" width="800"/>
</p>
<p align="center">
  <em>Figure: High-resolution dense features of MedDINOv3. We visualize the cosine similarity maps between the patches
marked with a red dot and all other patches. Input image at 2048 × 2048. </em>
</p>

## 📦 Pretrained Models

### 🔹 MedDINOv3 pretrained on CT-3M
| Backbone | Pretraining Dataset   | Download | Hugging Face |
|----------|-----------------------|----------|--------------|
| ViT-B/16 | CT-3M (3.87M slices) | [[Google Drive]](https://drive.google.com/file/d/1_MgctUnIIFcQJCVOhkcs84qq92hlqXVA/view?usp=sharing) | [[HF Model]](https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M) |

---
## ⚙️ Installation
### 1. 一些加速安装依赖的命令
在终端执行，配置conda安装依赖的国内镜像源
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

### 2. 创建环境
1. 创建新的 conda 环境（统一用一个）

```bash
conda create -n dino_biopsy python=3.10 -yconda activate dino_biopsy
```

2. 安装 PyTorch
有 NVIDIA GPU（推荐）：

```bash
conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```

3. 安装两个项目共有/额外依赖
```bash
pip install \
    monai[all] \
    natsort \
    pystrum \
    scipy \
    nibabel \
    matplotlib \
    plotly \
    tensorboard \
    omegaconf
```

然后 dinov3 一般还会用到一些常见库（如果你后面只做特征抽取/加载已有权重，这些也很有用）：

```bash
pip install \
    opencv-python \
    scikit-learn \
    pandas \
    tqdm \
    termcolor \
    psutil
```

### 3. 下载数据集并配置数据集的路径
在train和infer文件的前面找到下面两个变量然后更改为自己的路径,train在59行左右

    train_dir = '/home/gyl/DataSets/muProReg_process/train'
    val_dir = '/home/gyl/DataSets/muProReg_process/val'

infer修改test_dir，muProReg_process填val目录，Biopsy数据集有test目录
## 推理
inference目录下是一些MedDINO可视化图特征的一些推理代码，打开之后全部运行就可以
```
inference/demo.ipynb
```

##  训练


### 2. Locate the dinov3Trainer in 
```bash
nnUNet/nnunetv2/training/nnUNetTrainer/dinov3Trainer.py
```
Specify DINOv3 checkpoint paths in :
```python
@staticmethod
def build_network_architecture(patch_size: tuple, 
                                architecture_class_name: str,
                                arch_init_kwargs: dict,
                                arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:

    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_base
    # Initialize model
    model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05, n_storage_tokens=4, 
                        qkv_bias = False, mask_k_bias= True)
    # Load checkpoint
    chkpt = torch.load(
        YOUR_PATH_TO_CHECKPOINT,
        map_location='cpu'
    )
    state_dict = chkpt['teacher']
    state_dict = {
        k.replace('backbone.', ''): v
        for k, v in state_dict.items()
        if 'ibot' not in k and 'dino_head' not in k
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus import Primus_Multiscale
    primus = Primus_Multiscale(embed_dim=768, patch_embed_size=16, num_classes=num_output_channels, 
                                dino_encoder=model, interaction_indices=[2,5,8,11])
    return primus
```
### 3. 🚀 Training
#### Our MedDINOv3:
```bash
nnUNetv2_train dataset_id 2d 0 -tr meddinov3_base_primus_multiscale_Trainer
``` 

#### Training with original DINOv3 checkpoint:
```bash
nnUNetv2_train dataset_id 2d 0 -tr dinov3_base_primus_multiscale_Trainer
``` 

#### 2D nnUNet:
 ```bash
nnUNetv2_train dataset_id 2d 0
``` 

#### 2D SegFormer:
 ```bash
nnUNetv2_train dataset_id 2d 0 -tr segformerTrainer
``` 

#### 2D Dino UNet:
 ```bash
nnUNetv2_train dataset_id 2d 0 -tr dinoUNetTrainer
``` 

### 3. Calculate metrics:
 ```bash
python nnUNet/nnunetv2/compute_metrics.py
``` 

## 🌟 Tricks for you to reproduce our results
### Add a new 2d config so that the patch size is 896 x 896. 
Go to the nnUNetPlans.json, add something like this:
 ```
        "2d_896": {
            "inherits_from": "2d",
            "data_identifier": "nnUNetPlans_2d_896",
            "preprocessor_name": "DefaultPreprocessor",
            "patch_size": [
                896,
                896
            ],
            "spacing": [
                0.4464285714285714,
                0.4464285714285714
            ]
        }
```
Derive the spacing from the original 2d spacing. We use a formula like this:
```
new_spacing = original_spacing * (original_patch_size / 896)
```
Rerun the preprocessing
``` bash
nnUNetv2_plan_and_preprocess -d dataset_id -c 2d_896
```
## 📖 Citation

If you find this work useful, please cite:
``` 
@article{li2025meddinov3,
  title={MedDINOv3: How to adapt vision foundation models for medical image segmentation?},
  author={Li, Yuheng and Wu, Yizhou and Lai, Yuxiang and Hu, Mingzhe and Yang, Xiaofeng},
  journal={arXiv preprint arXiv:2509.02379},
  year={2025}
}
```
## 🙏 Acknowledgements

This project builds on:

nnU-Net https://github.com/MIC-DKFZ/nnUNet

DINOv3 https://github.com/facebookresearch/dinov3
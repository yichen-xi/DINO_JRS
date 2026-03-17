import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from torchvision import transforms
from sklearn.decomposition import PCA
import os, sys

# --------------------------
# 关键：配置Matplotlib后端，确保在PyCharm内部显示
# --------------------------
import matplotlib
# matplotlib.use('module://matplotlib_inline.backend_inline')  # 强制内嵌后端
# plt.rcParams['figure.figsize'] = (12, 6)  # 调整默认图像大小


# --------------------------
# 1. 读取.nii.gz格式的三维MRI数据
# --------------------------
mri_path = '/home/gyl/DataSets/muProReg_process/val/us_labels/case000000.nii.gz'
nifti_img = nib.load(mri_path)
mri_3d = nifti_img.get_fdata()  # (H, W, D) = (128, 128, 30)
print(f"三维MRI数据形状：{mri_3d.shape}")
# 前列腺miccai的label处理
mri_3d=mri_3d[...,0]
mri_3d=mri_3d.transpose((1,2,0))
# --------------------------
# 2. 提取第三维度的中间切片
# --------------------------
third_dim_axis = -1  # 最后一个轴为深度维度D
third_dim_size = mri_3d.shape[third_dim_axis]
middle_idx = third_dim_size // 2  # 15
print(f"第三维度大小：{third_dim_size}，中间切片索引：{middle_idx}")

# 提取二维切片
mri_slice = np.take(mri_3d, indices=middle_idx, axis=third_dim_axis)  # (128, 128)


# --------------------------
# 3. 数据预处理（适配ViT输入）
# --------------------------
def normalize_mri(img):
    """MRI切片归一化（避免极端值影响）"""
    img_tensor = torch.tensor(img).unsqueeze(0).float()  # (1, 128, 128)
    # 钳位到合理范围（根据MRI数据调整，此处用0-99百分位避免异常值）
    clamp_min = torch.quantile(img_tensor, 0.01)
    clamp_max = torch.quantile(img_tensor, 0.99)
    img_tensor = torch.clamp(img_tensor, clamp_min, clamp_max)
    # 标准化到0-1
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    return img_tensor


def make_transform(resize_size: int = 2048):
    """调整切片大小到ViT输入尺寸（2048x2048）"""
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size), antialias=True)  # 保持图像质量
    ])


# 执行预处理
normalized_slice = normalize_mri(mri_slice)  # (1, 128, 128)
resize_size = 512
resized_slice = make_transform(resize_size)(normalized_slice)  # (1, 2048, 2048)
resized_slice = resized_slice.repeat(3, 1, 1)  # 转为3通道（ViT默认输入3通道）
print(f"预处理后输入形状：{resized_slice.shape}")  # 输出：(3, 2048, 2048)


# --------------------------
# 4. 加载ViT模型（保持原逻辑）
# --------------------------
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
dinov3_parent = os.path.join(
    repo_root, "nnUNet", "nnunetv2", "training", "nnUNetTrainer", "dinov3"
)
for p in (dinov3_parent, repo_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from dinov3.models.vision_transformer import vit_base

# 初始化模型（注意n_storage_tokens=4，与原代码一致）
model = vit_base(
    drop_path_rate=0.2,
    layerscale_init=1.0e-05,
    n_storage_tokens=4,  # 关键：保留存储token配置
    qkv_bias=False,
    mask_k_bias=True
)

# 加载权重
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
chkpt_path = '/home/gyl/project/MedDINOv3/model/model.pth'
chkpt = torch.load(chkpt_path, weights_only=False, map_location='cpu')
state_dict = chkpt['teacher']
state_dict = {
    k.replace('backbone.', ''): v
    for k, v in state_dict.items()
    if 'ibot' not in k and 'dino_head' not in k
}
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"模型加载完成，设备：{device}")


# --------------------------
# 5. 修复后的PCA可视化函数（核心修改）
# --------------------------
def visualize_pca_layer(
    sample,
    model,
    input_size=2048,
    layer_indices=[2, 5, 8, 11],  # 要可视化的层索引
    device='cuda'
):
    # --------------------------
    # 关键1：计算ViT的patch数量和非patch token数量
    # --------------------------
    patch_size = model.patch_embed.proj.kernel_size[0]  # ViT默认patch_size=16
    num_patch = (input_size // patch_size) ** 2  # 128×128=16384（patch token数量）
    num_non_patch_token = 1 + model.n_storage_tokens  # 1(CLS) +4(存储)=5（非patch token数量）
    print(f"patch大小：{patch_size}，patch数量：{num_patch}，非patch token数量：{num_non_patch_token}")

    # --------------------------
    # 关键2：钩子函数捕获中间层特征（只保留patch token）
    # --------------------------
    intermediate_features = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            # 从列表中提取特征张量（适配原模型输出结构）
            feat_tensor = output[0] if isinstance(output, list) else output  # (1, num_tokens, feat_dim)
            # 剔除前N个非patch token（保留后面的patch token）
            patch_feat = feat_tensor[:, num_non_patch_token:, :]  # (1, 16384, feat_dim)
            intermediate_features[layer_idx] = patch_feat.clone()
        return hook

    # 注册钩子
    hooks = []
    for idx in layer_indices:
        if 0 <= idx < len(model.blocks):
            hook = model.blocks[idx].register_forward_hook(hook_fn(idx))
            hooks.append(hook)
        else:
            raise ValueError(f"层索引{idx}超出模型总层数{len(model.blocks)}")

    # --------------------------
    # 6. 前向传播（捕获特征）
    # --------------------------
    with torch.no_grad():
        # ViT输入需为(batch, channel, H, W)，此处batch=1
        input_tensor = sample.unsqueeze(0).to(device)  # (1, 3, 2048, 2048)
        _ = model(input_tensor, is_training=True)  # 触发钩子

    # 移除钩子（避免内存泄漏）
    for hook in hooks:
        hook.remove()

    # --------------------------
    # 7. PCA可视化（修复reshape逻辑）
    # --------------------------
    num_layers = len(layer_indices)
    # 创建子图：1行（原始图 + 每层PCA特征图）
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(6 * (num_layers + 1), 6))

    # 显示原始图像（归一化到0-1，便于查看）
    norm_sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
    axes[0].imshow(np.transpose(norm_sample.cpu().numpy(), (1, 2, 0)), cmap="gray")
    axes[0].axis('off')
    axes[0].set_title("Original MRI Slice")

    # 逐层可视化PCA特征
    for i, layer_idx in enumerate(layer_indices):
        # 获取当前层的patch特征（1, 16384, feat_dim）
        patch_feat = intermediate_features[layer_idx].squeeze(0).cpu().detach().numpy()  # (16384, feat_dim)

        # PCA降维（3维，便于彩色显示）
        pca = PCA(n_components=3, whiten=True)
        pca_feat = pca.fit_transform(patch_feat)  # (16384, 3)

        # 归一化到0-1（适配图像显示范围）
        norm_pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min() + 1e-8)

        # 关键3：正确reshape（16384 = 128×128，3为PCA维度）
        pca_img = norm_pca_feat.reshape(
            input_size // patch_size,  # 128（高）
            input_size // patch_size,  # 128（宽）
            3  # PCA降维后的3个通道
        )

        # 显示PCA特征图
        axes[i+1].imshow(pca_img)
        axes[i+1].axis('off')
        axes[i+1].set_title(f"Layer {layer_idx+1} (PCA Features)")

    # 确保图像在PyCharm内部显示
    plt.tight_layout()
    plt.show()


# --------------------------
# 8. 执行可视化（测试修复效果）
# --------------------------
if __name__ == "__main__":
    visualize_pca_layer(
        sample=resized_slice,  # (3, 2048, 2048)
        model=model,
        input_size=512,
        layer_indices=[0,1,2,3,4,5,6,7,8,9,10,11],  # 可同时可视化多层（如第1、2层）
        device=device
    )
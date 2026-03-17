import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from torchvision import transforms
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel  # 导入DINOv2工具


# --------------------------
# 1. 读取.nii.gz格式的三维MRI数据
# --------------------------
mri_path = '/home/gyl/DataSets/dataset/test/fixed_images/case_0301_01.nii.gz'
nifti_img = nib.load(mri_path)
mri_3d = nifti_img.get_fdata()  # 形状为 (H, W, D)
print(f"三维MRI数据形状：{mri_3d.shape}")


# --------------------------
# 2. 提取第三维度的中间切片
# --------------------------
third_dim_axis = -1  # 深度维度为最后一个轴
third_dim_size = mri_3d.shape[third_dim_axis]
middle_idx = third_dim_size // 2  # 中间切片索引
print(f"第三维度大小：{third_dim_size}，中间切片索引：{middle_idx}")
mri_slice = np.take(mri_3d, indices=middle_idx, axis=third_dim_axis)  # 提取二维切片 (H, W)


# --------------------------
# 3. 数据预处理（适配DINOv2输入）
# --------------------------
def normalize_mri(img):
    """MRI切片归一化（限制范围+标准化到0-1）"""
    img_tensor = torch.tensor(img).unsqueeze(0).float()  # 形状: (1, H, W)
    # 钳位到0-99百分位（过滤异常值）
    clamp_min = torch.quantile(img_tensor, 0.01)
    clamp_max = torch.quantile(img_tensor, 0.99)
    img_tensor = torch.clamp(img_tensor, clamp_min, clamp_max)
    # 标准化到0-1
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    return img_tensor

def make_transform(resize_size: int = 336):
    """调整切片到DINOv2输入尺寸（14的整数倍，如336=14×24）"""
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size), antialias=True)
    ])

normalized_slice = normalize_mri(mri_slice)  # 形状: (1, H, W)
resize_size = 14*128  # DINOv2输入尺寸（14×24）
resized_slice = make_transform(resize_size)(normalized_slice)  # 形状: (1, 336, 336)
resized_slice = resized_slice.repeat(3, 1, 1)  # 转为3通道（DINOv2默认输入3通道）
print(f"预处理后输入形状：{resized_slice.shape}")  # 预期: (3, 336, 336)


# --------------------------
# 4. 加载本地DINOv2模型
# --------------------------
local_model_path = "/home/gyl/models/dinov2-base"  # 本地模型路径
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载图像处理器和模型
image_processor = AutoImageProcessor.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)
model.to(device)
model.eval()  # 设为评估模式
print(f"DINOv2模型加载完成，运行设备：{device}")


# --------------------------
# 5. 适配DINOv2的PCA可视化函数
# --------------------------
def visualize_pca_layer(
    sample,
    model,
    input_size=336,
    layer_indices=[2, 5, 8, 11],  # 要可视化的层索引
    device='cuda'
):
    # 1. 计算Patch参数
    patch_size = 14  # DINOv2-base的Patch Size为14×14
    num_patch = (input_size // patch_size) ** 2  # 336//14=24 → 24×24=576个Patch
    num_non_patch_token = 1  # 仅1个CLS Token（无Storage Token）
    print(f"Patch大小：{patch_size}，Patch数量：{num_patch}，非Patch Token数量：{num_non_patch_token}")

    # 2. 钩子函数：捕获中间层Patch特征
    intermediate_features = {}
    def hook_fn(layer_idx):
        def hook(module, input, output):
            feat_tensor = output[0]  # DINOv2输出为 (batch, num_tokens, hidden_dim)
            patch_feat = feat_tensor[ num_non_patch_token:, :]  # 剔除CLS Token，保留Patch Token
            intermediate_features[layer_idx] = patch_feat.clone()
        return hook

    # 3. 注册钩子到Transformer层
    hooks = []
    for idx in layer_indices:
        if 0 <= idx < len(model.encoder.layer):
            hook = model.encoder.layer[idx].register_forward_hook(hook_fn(idx))
            hooks.append(hook)
        else:
            raise ValueError(f"层索引{idx}超出范围，模型共{len(model.encoder.layer)}层")

    # 4. 前向传播（触发钩子，捕获特征）
    with torch.no_grad():
        input_tensor = sample.unsqueeze(0).to(device)  # 形状: (1, 3, 336, 336)
        _ = model(input_tensor, output_hidden_states=True)  # 需输出所有层的hidden states

    # 移除钩子（避免内存泄漏）
    for hook in hooks:
        hook.remove()

    # 5. PCA可视化
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(6 * (num_layers + 1), 6))

    # 显示原始图像
    norm_sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
    axes[0].imshow(np.transpose(norm_sample.cpu().numpy(), (1, 2, 0)), cmap="gray")
    axes[0].axis('off')
    axes[0].set_title("Original MRI Slice")

    # 逐层可视化PCA特征
    for i, layer_idx in enumerate(layer_indices):
        patch_feat = intermediate_features[layer_idx].squeeze(0).cpu().detach().numpy()  # (576, hidden_dim)

        # PCA降维到3维（用于RGB显示）
        pca = PCA(n_components=3, whiten=True)
        pca_feat = pca.fit_transform(patch_feat)

        # 归一化到0-1范围
        norm_pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min() + 1e-8)

        # 重塑为图像形状: (24, 24, 3)
        pca_img = norm_pca_feat.reshape(
            input_size // patch_size,
            input_size // patch_size,
            3
        )

        axes[i+1].imshow(pca_img)
        axes[i+1].axis('off')
        axes[i+1].set_title(f"Layer {layer_idx+1} (PCA Features)")

    plt.tight_layout()
    plt.show()


# --------------------------
# 6. 执行可视化
# --------------------------
if __name__ == "__main__":
    visualize_pca_layer(
        sample=resized_slice,
        model=model,
        input_size=resize_size,
        layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 可视化所有12层
        device=device
    )
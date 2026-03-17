import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle  # 新增：用于读取.pkl文件
from torchvision import transforms
from sklearn.decomposition import PCA
import os, sys

# --------------------------
# 关键：配置Matplotlib后端，确保在PyCharm内部显示
# --------------------------
import matplotlib

# matplotlib.use('module://matplotlib_inline.backend_inline')  # 强制内嵌后端（注释会导致不显示）
# plt.rcParams['figure.figsize'] = (12, 6)  # 调整默认图像大小

# --------------------------
# 1. 读取.pkl格式的三维数据（核心修改部分）
# --------------------------
pkl_path = '/home/gyl/DataSets/IXI_data_half/Test/subject_1.pkl'  # .pkl文件路径

# 读取.pkl文件（根据你的.pkl数据结构，二选一或调整）
with open(pkl_path, 'rb') as f:
    # 情况1：.pkl直接存储三维numpy数组（形状 (H, W, D)）
    # （最常见情况，若你的.pkl是这种结构，保留这行，注释情况2）
    mri_3d,_ = pickle.load(f)

    # 情况2：.pkl存储字典（如 {'image': 三维数组, 'label': ...}）
    # （若你的.pkl是字典，需替换'image'为实际的键名，删除情况1的代码）
    # pkl_data = pickle.load(f)
    # mri_3d = pkl_data['image']  # 关键：替换为字典中存储图像的键名（如'data'/'img'等）

# 验证读取的数据格式（必须是三维numpy数组）
assert isinstance(mri_3d, np.ndarray), f".pkl读取结果不是numpy数组，而是{type(mri_3d)}"
assert mri_3d.ndim == 3, f".pkl数据不是三维数组，维度为{mri_3d.ndim}"
print(f"三维.pkl数据形状：{mri_3d.shape} → 格式要求：(H, W, D) 或 (x, y, z)")

# --------------------------
# 2. 提取第三维度的中间切片（完全不变）
# --------------------------
third_dim_axis = -3  # 最后一个轴为深度维度D
third_dim_size = mri_3d.shape[third_dim_axis]
middle_idx = third_dim_size // 2  # 中间切片索引
print(f"第三维度大小：{third_dim_size}，中间切片索引：{middle_idx}")

# 提取二维切片
mri_slice = np.take(mri_3d, indices=middle_idx, axis=third_dim_axis)  # (H, W)


# --------------------------
# 3. 数据预处理（适配ViT输入，完全不变）
# --------------------------
def normalize_mri(img):
    """MRI切片归一化（避免极端值影响）"""
    img_tensor = torch.tensor(img).unsqueeze(0).float()  # (1, H, W)
    # 钳位到合理范围（0-99百分位，避免异常值）
    clamp_min = torch.quantile(img_tensor, 0.01)
    clamp_max = torch.quantile(img_tensor, 0.99)
    img_tensor = torch.clamp(img_tensor, clamp_min, clamp_max)
    # 标准化到0-1（便于后续显示和模型输入）
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    return img_tensor


def make_transform(resize_size: int = 2048):
    """调整切片大小到ViT输入尺寸（2048x2048）"""
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size), antialias=True)  # 保持图像质量
    ])


# 执行预处理
normalized_slice = normalize_mri(mri_slice)  # (1, H, W)
resize_size = 512
resized_slice = make_transform(resize_size)(normalized_slice)  # (1, 2048, 2048)
resized_slice = resized_slice.repeat(3, 1, 1)  # 转为3通道（ViT默认输入3通道）
print(f"预处理后输入形状：{resized_slice.shape}")  # 预期输出：(3, 2048, 2048)

# --------------------------
# 4. 加载ViT模型（完全不变）
# --------------------------
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
dinov3_parent = os.path.join(
    repo_root, "nnUNet", "nnunetv2", "training", "nnUNetTrainer", "dinov3"
)
for p in (dinov3_parent, repo_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from dinov3.models.vision_transformer import vit_base

# 初始化模型（保持n_storage_tokens=4的配置）
model = vit_base(
    drop_path_rate=0.2,
    layerscale_init=1.0e-05,
    n_storage_tokens=4,
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
# 5. PCA可视化函数（完全不变）
# --------------------------
def visualize_pca_layer(
        sample,
        model,
        input_size=2048,
        layer_indices=[2, 5, 8, 11],  # 可调整要可视化的层
        device='cuda'
):
    # 计算patch数量和非patch token数量
    patch_size = model.patch_embed.proj.kernel_size[0]
    num_patch = (input_size // patch_size) ** 2
    num_non_patch_token = 1 + model.n_storage_tokens
    print(f"patch大小：{patch_size}，patch数量：{num_patch}，非patch token数量：{num_non_patch_token}")

    # 钩子函数捕获中间层特征（只保留patch token）
    intermediate_features = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            feat_tensor = output[0] if isinstance(output, list) else output
            patch_feat = feat_tensor[:, num_non_patch_token:, :]
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

    # 前向传播捕获特征
    with torch.no_grad():
        input_tensor = sample.unsqueeze(0).to(device)
        _ = model(input_tensor, is_training=True)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 可视化
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(6 * (num_layers + 1), 6))

    # 显示原始图像
    norm_sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
    axes[0].imshow(np.transpose(norm_sample.cpu().numpy(), (1, 2, 0)), cmap="gray")
    axes[0].axis('off')
    axes[0].set_title("Original Image (from .pkl)")

    # 逐层显示PCA特征
    for i, layer_idx in enumerate(layer_indices):
        patch_feat = intermediate_features[layer_idx].squeeze(0).cpu().detach().numpy()
        # PCA降维
        pca = PCA(n_components=3, whiten=True)
        pca_feat = pca.fit_transform(patch_feat)
        # 归一化
        norm_pca_feat = (pca_feat - pca_feat.min()) / (pca_feat.max() - pca_feat.min() + 1e-8)
        # 重塑为图像格式
        pca_img = norm_pca_feat.reshape(
            input_size // patch_size,
            input_size // patch_size,
            3
        )
        # 显示
        axes[i + 1].imshow(pca_img)
        axes[i + 1].axis('off')
        axes[i + 1].set_title(f"Layer {layer_idx + 1} (PCA)")

    # 确保在PyCharm内部显示
    plt.tight_layout()
    plt.show()


# --------------------------
# 6. 执行可视化（完全不变）
# --------------------------
if __name__ == "__main__":
    visualize_pca_layer(
        sample=resized_slice,
        model=model,
        input_size=resize_size,
        layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 可视化所有12层
        device=device
    )
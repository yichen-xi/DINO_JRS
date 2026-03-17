

# 1. 定义切片和预处理函数（无修改，确保输入尺寸正确）
def slice_3d_mri(mri_tensor):
    """向量化切片：直接拆分最后一个维度，返回[D,1,H,W]的批量张量（无需列表）"""
    # 原逻辑：for循环逐个切片 → 列表
    # 优化后：用unbind拆分dim=4，再stack成[D,1,H,W]
    slices = torch.stack(torch.unbind(mri_tensor, dim=4), dim=0)  # [D,1,H,W]
    return slices


import torch.nn.functional as F  # 用PyTorch的functional接口，支持GPU张量


def preprocess_slice_batch(slice_batch, target_size=(224, 224)):
    B, C, H, W = slice_batch.shape
    device = slice_batch.device

    # 1. 归一化到[0,255]（合并维度计算，减少view操作）
    # 原逻辑：view(B,1,-1) → 优化为flatten(2)，更高效
    min_val = slice_batch.flatten(2).min(dim=2, keepdim=True)[0].unsqueeze(-1)  # [B,1,1,1,1]
    max_val = slice_batch.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1)
    slice_norm = (slice_batch - min_val) / (max_val - min_val + 1e-8) * 255.0

    # 2. 移除uint8转换（DINOv3可接受float32输入，避免类型转换开销）
    # 原逻辑：to(torch.uint8) → 直接用float32，减少转换
    slice_norm = slice_norm.clamp(0, 255)  # 确保值在合理范围

    # 3. 批量Resize（用F.interpolate替代F.resize，支持批量优化）
    slice_resized = F.interpolate(
        slice_norm,
        size=target_size,
        mode='bilinear',
        align_corners=True
    )  # [B,1,224,224]

    # 4. 单通道转3通道（用repeat替代cat，更高效）
    slice_3ch = slice_resized.repeat(1, 3, 1, 1)  # [B,3,224,224]

    return slice_3ch  # 无需再转float（已为float32）


import torch
import torch.nn as nn
import os
import sys
# 注意：DINOv3 需从本地导入模型，而非 Hugging Face AutoModel
from dinov3.models.vision_transformer import vit_base  # 本地DINOv3模型类


class DINOv3MultiLayerExtractor(nn.Module):
    def __init__(self,
                 dinov3_model_path="/home/gyl/project/MedDINOv3/model/model.pth",  # DINOv3权重路径
                 dinov3_parent_dir="/home/gyl/project/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3",  # DINOv3代码根目录
                 feat_dim=768,  # DINOv3-base 隐藏层维度（固定768）
                 reduce_dim=128,
                 n_storage_tokens=4,
                 target_layers=[0,1,2,3]):  # DINOv3特色：存储Token数量（默认4，需与权重一致）
        super().__init__()
        # --------------------------
        # 1. 配置DINOv3模型路径（关键：确保能导入vit_base）
        # --------------------------
        if dinov3_parent_dir not in sys.path:
            sys.path.insert(0, dinov3_parent_dir)  # 加入Python搜索路径

        # --------------------------
        # 2. 初始化DINOv3模型结构（匹配权重训练配置）
        # --------------------------
        self.model = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1.0e-05,
            n_storage_tokens=n_storage_tokens,  # 必须与权重的存储Token数一致
            qkv_bias=False,
            mask_k_bias=True
        )

        # --------------------------
        # 3. 加载并适配DINOv3预训练权重（教师模型骨干权重）
        # --------------------------
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        chkpt = torch.load(dinov3_model_path, weights_only=False, map_location="cpu")
        state_dict = chkpt["teacher"]  # 提取教师模型权重（DINOv3用蒸馏训练）
        # 适配权重键名：去掉"backbone."前缀，过滤训练专用头（ibot/dino_head）
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if "ibot" not in k and "dino_head" not in k
        }
        self.model.load_state_dict(state_dict)  # 加载权重
        self.model.to(device)  # 移到目标设备
        self.model.eval()  # 推理模式：固定BatchNorm/Dropout

        # --------------------------
        # 4. 特征提取配置（继承原逻辑，适配DINOv3 Token）
        # --------------------------
        self.feat_dim = feat_dim
        self.reduce_dim = reduce_dim
        self.n_storage_tokens = n_storage_tokens  # 记录非Patch Token数量（1CLS+4存储=5）
        self.target_layers = target_layers  # 要提取的中间层（可调整）
        self.mid_features = []  # 保存中间层特征

        # --------------------------
        # 5. 注册钩子（目标层：DINOv3的model.blocks，而非DINOv2的model.encoder.layer）
        # --------------------------
        # self.hooks = []
        # for layer_idx in self.target_layers:
        #     # DINOv3的Transformer层在model.blocks，DINOv2在model.encoder.layer
        #     hook = self.model.blocks[layer_idx].register_forward_hook(
        #         self._save_mid_feature
        #     )
        #     self.hooks.append(hook)

        # --------------------------
        # 6. 降维层（与原逻辑一致，适配设备）
        # --------------------------
        self.reduce_layers = nn.ModuleList()
        for i in range(len(self.target_layers)):
            reduce_conv = nn.Conv2d(
                feat_dim,
                reduce_dim // (2 ** (3 - i)),  # 原降维逻辑：128→64→32→16（4层）
                kernel_size=1
            )
            self.reduce_layers.append(reduce_conv.to(device))  # 移到模型设备
    #
    # def _save_mid_feature(self, module, input, output):
    #     """钩子回调：捕获DINOv3中间层特征（形状：[B, num_tokens, 768]）"""
    #     # DINOv3的blocks输出为 (token_feat, 辅助信息)，取第一个元素（token特征）
    #     feat_tensor = output[0] if isinstance(output, list) else output
    #     self.mid_features.append(feat_tensor)

    def forward(self, slice_batch):
        """
        批量提取DINOv3特征（接口与原DINOv2函数一致）
        input: slice_batch - [D, 3, H, W]（D=切片数，H/W需为16的整数倍，如224/16=14）
        output: 4个层的特征列表，每个特征形状 [D, reduce_dim//2^(3-i), H_patch, W_patch]
        """
        # self.mid_features.clear()  # 清空历史特征
        D = slice_batch.shape[0]  # 批量大小（切片数）
        device = slice_batch.device

        # --------------------------
        # 7. 前向传播（DINOv3需指定is_training=True，确保输出token特征）
        # --------------------------
        with torch.no_grad():
            slice_batch = slice_batch.to(device)  # 确保设备一致
            # DINOv3的forward需要is_training参数（DINOv2不需要）
            # _ = self.model(slice_batch, is_training=True)
            mid_features=self.model.get_features_front_n(slice_batch,n=self.target_layers,norm=True,reshape=True)
        # --------------------------
        # 8. 特征处理（核心差异：剔除5个非Patch Token，而非DINOv2的1个）
        # --------------------------
        processed_features = []
        for i, feat in enumerate(mid_features):
            # ① 剔除非Patch Token：1个CLS + 4个存储Token = 5个，取[:,5:,:]
            # DINOv2仅剔除1个CLS（[:,1:,:]），DINOv3需剔除5个
            patch_feat = feat  # [D, num_patch, 768]

            # ② 动态计算Patch空间尺寸（如224→224/16=14 → 14×14=196个Patch）
            # num_patches = patch_feat.shape[1]
            #
            # # ③ 重塑为空间特征：[D, 768, h_patch, w_patch]（与原逻辑一致）
            # spatial_feat = patch_feat.permute(0, 2, 1).reshape(
            #     D, self.feat_dim, h_patch, w_patch
            # )

            # ④ 降维（与原逻辑一致）
            reduced_feat = self.reduce_layers[i](feat)
            processed_features.append(reduced_feat)

        return processed_features

    # def __del__(self):
    #     """销毁时移除钩子，避免内存泄漏（与原逻辑一致）"""
    #     for hook in self.hooks:
    #         hook.remove()


# 3. 上采样模块（FP32，适配批量特征）
class UpBlock(nn.Module):
    """空间上采样+通道调整（FP32，支持批量输入）"""

    def __init__(self,  target_size,ch):
        super().__init__()
        self.target_size = target_size  # 目标尺寸，如(80,96)
        self.up = nn.Sequential(
            # 1. 1×1卷积：调整通道数（FP32）
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            # nn.ReLU(inplace=True),  # 原地激活，节省显存
            # 2. 双线性插值：上采样到目标尺寸（保留细节，无伪影）
            nn.Upsample(
                size=target_size,
                mode='bilinear',
                align_corners=True  # 对齐角落像素，提升精度
            ),
            # 3. 3×3卷积：修复插值模糊，增强空间特征
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(ch)
        )
        # 上采样模块移到GPU
        # self.up = self.up.to(next(self.parameters()).device if hasattr(self, 'parameters') else 'cuda')

    def forward(self, x):
        """
        input: x - [D, in_channels, H_patch, W_patch]（批量特征）
        output: [D, out_channels, target_H, target_W]（上采样后特征）
        """
        return self.up(x)


# 在降维的时候一次性降维到想要的维度
class MRIMultiLayerFusion(nn.Module):
    def __init__(self, reduce_dim=16,inshape=(128,128,32),target_size=(16*40, 16*48),target_layers=[0,1,2,3]):
        super().__init__()
        # 初始化DINOv2特征提取器（reduce_dim*8=128，与原逻辑一致）
        self.inshape = inshape
        self.target_size = target_size
        self.feature_extractor = DINOv3MultiLayerExtractor(
            reduce_dim=reduce_dim * 8,
            feat_dim=768,
            target_layers=target_layers
        )
        self.device = next(self.feature_extractor.model.parameters()).device  # 统一设备
        self.reduce_dim = reduce_dim

        # 1. 深度下采样池化（仅降维深度维度，保留空间维度）
        self.depth_pools = nn.ModuleList([
            nn.AvgPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1)),  # 32→32
            nn.AvgPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)),  # 32→16
            nn.AvgPool3d(kernel_size=(1, 1, 4), stride=(1, 1, 4)),  # 32→8
            nn.AvgPool3d(kernel_size=(1, 1, 8), stride=(1, 1, 8))  # 32→4
        ])
        # 各层输出通道数（与原逻辑一致：16,32,64,128）
        self.reduce_dims = [reduce_dim, reduce_dim * 2, reduce_dim * 4, reduce_dim * 8]

        # 2. 3D卷积融合层（FP32，适配各层通道数）
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(d, d, kernel_size=3, padding=1),
                nn.InstanceNorm3d(d),
                nn.LeakyReLU(inplace=True),
            ) for d in self.reduce_dims
        ])
        # 3D卷积移到GPU
        self.fusion_convs = self.fusion_convs.to(self.device)

        # 3. 上采样模块（适配各层目标尺寸：80×96、40×48、20×24、10×12）
        self.up_blocks = nn.ModuleList()
        for i in range(4):
            self.up_blocks.append(UpBlock(ch=reduce_dim*2**i,target_size=(inshape[0]//2**i, inshape[1]//2**i)))

    def forward(self, mri_3d):
        """
        批量处理3D MRI（FP32精度）
        input: mri_3d - [B, 1, H, W, D]（B=1，如[1,1,80,96,112]）
        output: 4个层的融合特征，形状分别为：
                [1,16,80,96,32], [1,32,40,48,16], [1,64,20,24,8], [1,128,10,12,4]
        """
        # 确保输入在GPU上（FP32）
        mri_3d = mri_3d.cuda()
        B, C, H, W, D = mri_3d.shape  # B=1, C=1, H=80, W=96, D=112（切片数）

        # 步骤1：批量切片 → [D,1,H,W]（将深度维度转为批量维度）
        # permute(0,4,1,2,3) → [1,112,1,80,96]，再reshape为[112,1,80,96]
        slice_batch = mri_3d.permute(0, 4, 1, 2, 3).reshape(D, 1, H, W)

        # 步骤2：批量预处理 → [D,3,224,224]（FP32，GPU）
        processed_batch = preprocess_slice_batch(slice_batch, target_size=self.target_size)

        # 步骤3：批量提取DINOv2特征 → 4个层，每个[D,128,16,16]
        batch_4feats = self.feature_extractor(processed_batch)

        # 步骤4：批量上采样+添加深度维度（无for循环，批量处理）
        layer_features = []
        for i in range(4):
            # 上采样：[D,128,16,16] → [D, out_c, target_H, target_W]
            up_feat = self.up_blocks[i](batch_4feats[i])
            # 添加深度维度（1个切片对应1个深度位置）：[D, out_c, H, W] → [D, out_c, H, W, 1]
            feat_with_depth = up_feat.unsqueeze(-1)
            layer_features.append(feat_with_depth)

        # 步骤5：3D融合+深度下采样（批量处理）
        final_features = []
        for i in range(4):
            # 1. 拼接深度维度：[D, c, H, W, 1] → [1, c, H, W, D]（B=1）
            # permute(1,2,3,0,4) → [c, H, W, D, 1]，reshape为[1,c,H,W,D]
            stacked_3d = layer_features[i].permute(4,1, 2, 3, 0) # 112,128,80,96,1 -> 1,128,80,96,112

            # 2. 适配3D卷积输入格式：[1, c, D, H, W]（3D卷积要求通道后接深度）
            # stacked_3d = stacked_3d.permute(0, 1, 4, 2, 3)

            # 3. 3D融合（FP32）
            fused = self.fusion_convs[i](stacked_3d)

            # 4. 深度下采样（如32→4）
            downsampled = self.depth_pools[i](fused)

            # 5. 还原维度顺序：[1, c, H, W, D_down]（输出格式）
            final_feat = downsampled
            final_features.append(final_feat)

        return final_features  # 返回4个层的FP32特征


#
if __name__ == "__main__":
    # 生成模拟3D MRI数据 [1,1,128,128,32]
    mri_3d = torch.randn(1, 1, 80, 96, 112).cuda()

    # 初始化融合模型
    fusion_model = MRIMultiLayerFusion(inshape=(80,96,112),target_size=(16*40,16*48)).cuda()

    # 前向传播（禁用梯度，节省显存）
    with torch.no_grad():
        layer2_feat, layer5_feat, layer8_feat, layer11_feat = fusion_model(mri_3d)

    # 打印每个层的特征形状（预期最后一维分别为32、16、8、4）
    print("输入MRI形状:", mri_3d.shape)
    print("层2特征形状（最后一维32）:", layer2_feat.shape)  # [1,64,16,16,32]（16是动态计算的patch尺寸）
    print("层5特征形状（最后一维16）:", layer5_feat.shape)  # [1,64,16,16,16]
    print("层8特征形状（最后一维8）:", layer8_feat.shape)  # [1,64,16,16,8]
    print("层11特征形状（最后一维4）:", layer11_feat.shape)  # [1,64,16,16,4]
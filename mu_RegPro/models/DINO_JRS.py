import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint
# from Biopsy.models.VLM_JRS_v2 import EncEnhanceBlock_attn


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the models files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class Reg_Head(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.defconv = nn.Conv3d(in_ch, 3, 3, 1, 1)
        self.defconv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv.weight.shape))
        self.defconv.bias = nn.Parameter(torch.zeros(self.defconv.bias.shape))

    def forward(self, x):
        flow=self.defconv(x)
        return flow

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha,inplace=True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvInsBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d((2*in_channels)//3, (2*in_channels)//3, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha,inplace=True),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha,inplace=True),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)

class RegEncoder(nn.Module):
    """
    Main models
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(RegEncoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]
class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            DoubleConv(c,c)
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha,inplace=True)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class SegEncoder(nn.Module):
    def __init__(self, n_channels,base_channels=16,p_drop=0.1):
        super(SegEncoder, self).__init__()
        self.n_channels = n_channels

        c=base_channels
        # 4 层 encoder，对应 4 个尺度的特征
        self.channels = [c, 2*c, 4*c, 8*c]

        # 初始卷积
        self.inc = ConvInsBlock(n_channels, self.channels[0])

        # 编码器
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])

        # self.drop2 = nn.Dropout3d(p=p_drop)      # 中层
        # self.drop3 = nn.Dropout3d(p=p_drop)      # 中层
        # self.drop4 = nn.Dropout3d(p=p_drop)      # 中层
        # self.drop5 = nn.Dropout3d(p=min(p_drop*2, 0.5))  # 瓶颈层


    def forward(self, x):
        """
        返回 4 个尺度的特征:
        x1: C,   x2: 2C, x3: 4C, x4: 8C
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        enc = [x1, x2, x3, x4]
        return enc
class ConvFusion(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=1,stride=1,padding=0):
        super().__init__()
        self.conv = ConvInsBlock(in_ch,out_ch,kernel_size,stride,padding)

    def forward(self,x,y):
        x=torch.cat([x,y],1)
        x=self.conv(x)
        return x

class feature_fusion(nn.Module):
    def __init__(self,ch=16):
        super().__init__()
        self.gates=nn.ModuleList()
        for i in range(4):
            self.gates.append(ConvFusion(2*ch*2**i,ch*2**i,1,1,0))

    def forward(self,enc,dino_enc):
        for i in range(4):
            enc[i]=self.gates[i](enc[i],dino_enc[i])
        return enc


class RegDecoder(nn.Module):
    """
    配准 decoder：
    - 输入：来自 RegEncoder 的 mov / fix 两路 encoder 特征（各 4 个尺度）
    - 逻辑：先在每个尺度上拼接 mov / fix 特征，然后走 UNet 式上采样，
            最终输出 3 通道形变场。
    """
    def __init__(self, base_channels=16, bilinear=False):
        super(RegDecoder, self).__init__()

        # RegEncoder 输出通道为 [C, 2C, 4C, 8C]
        # mov / fix 在每个尺度拼接后通道翻倍： [2C, 4C, 8C, 16C]
        c = base_channels
        self.channels = [2 * c, 4 * c, 8 * c, 16 * c]

        # 上采样模块，保持 “decoder_feat 通道 = 2 * encoder_feat 通道” 的关系
        # 这样与上面的 Up(in_channels, out_channels) 设计是匹配的
        self.up1 = Up(self.channels[3] + self.channels[2], self.channels[2], bilinear)  # 16C -> 8C
        self.up2 = Up(self.channels[2] + self.channels[1], self.channels[1], bilinear)  # 8C  -> 4C
        self.up3 = Up(self.channels[1] + self.channels[0], self.channels[0], bilinear)  # 4C  -> 2C

        # 最终使用最高分辨率的融合特征 (2C) 预测 3 通道形变场
        self.RegHead = Reg_Head(self.channels[0])

    def forward(self, enc_m, enc_f):
        """
        enc_m / enc_f: list[Tensor]，长度为 4，对应 4 个尺度特征
        每个 Tensor 形状为 [B, C_i, D, H, W]
        """
        assert len(enc_m) == 4 and len(enc_f) == 4, "RegDecoder 期望 4 层 encoder 特征"

        # 在每个尺度上拼接 mov / fix 特征
        x1 = torch.cat([enc_m[0], enc_f[0]], dim=1)  # 2C
        x2 = torch.cat([enc_m[1], enc_f[1]], dim=1)  # 4C
        x3 = torch.cat([enc_m[2], enc_f[2]], dim=1)  # 8C
        x4 = torch.cat([enc_m[3], enc_f[3]], dim=1)  # 16C

        # 自底向上的解码
        dec3 = self.up1(x4, x3)   # 16C -> 8C
        dec2 = self.up2(dec3, x2) # 8C  -> 4C
        dec1 = self.up3(dec2, x1) # 4C  -> 2C

        # 输出形变场
        flow = self.RegHead(dec1)
        return flow






class SegDecoder(nn.Module):
    """
    分割 decoder：
    - 输入：来自 SegEncoder 的 4 层特征 [C, 2C, 4C, 8C]
    - 输出：n_classes 通道的 segmentation logits
    """
    def __init__(self, base_channels, n_classes, bilinear=False):
        super(SegDecoder, self).__init__()
        self.base_channels = base_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        c = base_channels
        # SegEncoder 输出通道：[C, 2C, 4C, 8C]
        self.channels = [c, 2 * c, 4 * c, 8 * c]

        # 解码部分，同样满足 “decoder_feat 通道 = 2 * encoder_feat 通道”
        self.up1 = Up(self.channels[3] + self.channels[2], self.channels[2], bilinear)  # 8C -> 4C
        self.up2 = Up(self.channels[2] + self.channels[1], self.channels[1], bilinear)  # 4C -> 2C
        self.up3 = Up(self.channels[1] + self.channels[0], self.channels[0], bilinear)  # 2C -> C

        # 输出层：C -> n_classes
        self.outc = nn.Sequential(
            nn.Conv3d(self.channels[0], n_classes, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: list[Tensor]，长度为 4，对应 [C, 2C, 4C, 8C] 四个尺度
        """
        x1, x2, x3, x4 = x

        # 解码过程：自底向上融合 skip-connection
        dec3 = self.up1(x4, x3)   # 8C -> 4C
        dec2 = self.up2(dec3, x2) # 4C -> 2C
        dec1 = self.up3(dec2, x1) # 2C -> C

        # 输出 segmentation
        logits = self.outc(dec1)
        return logits

# 将v14的分割decoder特征和配准特征融合的模块换成LSKA_v3,
# 将原本的五层改成四层
from Biopsy.models.dinov3_encoder_v4_speedup_v4 import MRIMultiLayerFusion  # 直接导入需要的类，避免冗余
class dinov3_JRS(nn.Module):
    id=0
    def __init__(self, inshape=(128,128,32),in_channel=1,ch=16,bilinear=False,n_classes=1,target_size=(256, 256),
                 target_layers=[0,1,2,3]):
        super(dinov3_JRS, self).__init__()
        self.SegEncoder = SegEncoder(in_channel, ch)
        self.RegEncoder = SegEncoder(in_channel, ch)
        # self.EncoderEnc = EncEnhanceNet(ch)
        self.dino_encoder = MRIMultiLayerFusion(reduce_dim=ch, inshape=inshape, target_size=target_size,
                                                target_layers=target_layers)
        self.dino_fuse_reg = feature_fusion(ch)
        self.dino_fuse_seg = feature_fusion(ch)
        c = ch
        self.channels = [ch, 2 * ch, 4 * ch, 8 * ch]

        self.warp = SpatialTransformer(inshape)

        # bottleNeck
        self.RegDecoder = RegDecoder(c, bilinear)
        self.SegDecoder = SegDecoder(c, n_classes, bilinear)

    def forward(self, mov, fix):
        dino_m = self.dino_encoder(mov)
        dino_f = self.dino_encoder(fix)
        seg_enc_m = self.SegEncoder(mov)

        seg_enc_f = self.SegEncoder(fix)

        reg_enc_m = self.RegEncoder(mov)
        reg_enc_f = self.RegEncoder(fix)

        # reg_enc_m,seg_enc_m = self.EncoderEnc(reg_enc_m,seg_enc_m)
        # reg_enc_f,seg_enc_f = self.EncoderEnc(reg_enc_f,seg_enc_f)

        reg_enc_m = self.dino_fuse_reg(reg_enc_m, dino_m)
        reg_enc_f = self.dino_fuse_reg(reg_enc_f, dino_f)

        seg_enc_m = self.dino_fuse_seg(seg_enc_m, dino_m)
        seg_enc_f = self.dino_fuse_seg(seg_enc_f, dino_f)

        # 配准：使用 mov / fix 两路 encoder 特征共同预测 flow
        flow = self.RegDecoder(reg_enc_m, reg_enc_f)

        logits_x = self.SegDecoder(seg_enc_m)
        logits_y = self.SegDecoder(seg_enc_f)

        moved_img=self.warp(mov,flow)
        return moved_img,moved_img,flow,flow,logits_x,logits_y

if __name__ == '__main__':
    size = (1, 1, 80, 96, 112)
    model = dinov3_JRS(inshape=size[2:],n_classes=29).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    total_params_dinov3 = sum(p.numel() for p in model.dino_encoder.parameters())
    print(f"总参数数量: {total_params:,}")
    print(f"dinov3总参数数量: {total_params_dinov3:,}")
    # print(str(models))
    A = torch.ones(size).cuda()
    B = torch.ones(size).cuda()
    AL = torch.ones(size)
    BL = torch.ones(size)
    # print(model)

    moved_img,warped_fix,flow,flow_inv,logits_x,logits_y = model(A, B)
    print(moved_img.shape, flow.shape, logits_x.shape)

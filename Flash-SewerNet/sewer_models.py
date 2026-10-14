# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import os
import copy
#from mmcv.cnn import build_norm_layer
from math import log
import numpy
#aimport matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import math
import torch.nn.functional as F



def build_norm_layer(cfg, num_features):
    if cfg['type'] == 'BN':
        return 'bn', nn.BatchNorm2d(num_features)  # 返回 (name, layer)
    elif cfg['type'] == 'LN':
        return 'ln', nn.LayerNorm(num_features)
    else:
        raise ValueError(f"Unsupported norm type: {cfg['type']}")



class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, norm_layer, act_layer, feature_extra=False)
        self.norm_g = build_norm_layer(norm_layer, self.outdim)[1]

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x

def show_feature1(out):
    out_cpu = out.cpu()
    feature_map = out_cpu.detach().numpy()
    # [N， C, H, W] -> [H, W， C]
    im = numpy.squeeze(feature_map)
    im = numpy.transpose(im, [1, 2, 0])
    for c in range(24):
        ax = plt.subplot(4, 6, c + 1)
        plt.axis('off')  # 不显示坐标轴
        # [H, W, C]
        plt.imshow(im[:, :, c], cmap=plt.get_cmap('Blues'))
    plt.show()


def show_feature(out, max_channels=24):
    # Detach and convert to numpy
    out_cpu = out.detach().cpu()

    # Handle batch dimension (if present)
    if len(out_cpu.shape) == 4:  # [B, C, H, W]
        feature_map = out_cpu[0]  # Take first sample in batch
    elif len(out_cpu.shape) == 3:  # [C, H, W]
        feature_map = out_cpu
    else:
        raise ValueError(f"Unexpected feature map shape: {out_cpu.shape}")

    # Transpose to [H, W, C]
    im = np.transpose(feature_map.numpy(), [1, 2, 0])

    # Determine number of channels to show
    num_channels = min(im.shape[-1], max_channels)
    rows = math.ceil(num_channels / 6)

    # Plot each channel
    plt.figure(figsize=(15, 2 * rows))
    for c in range(num_channels):
        ax = plt.subplot(rows, 6, c + 1)
        plt.axis('off')
        plt.imshow(im[:, :, c], cmap='viridis')  # Using viridis for better visibility
    plt.tight_layout()
    plt.show()
    plt.pause(5)  # 暂停5秒


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])

    def forward(self, x):
        out = self.block(x)
        return out


class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class LFEA(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(LFEA, self).__init__()
        self.channel = channel
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1

        # 3x3卷积块
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer())

        # 通道注意力相关层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 归一化层
        self.norm = build_norm_layer(norm_layer, channel)[1]

        # 修改：使用动态输入维度
        self.u_proj = nn.Linear(512, channel)  # 假设CLIP输出是512维

    def forward(self, c: Tensor, att: Tensor, query: Tensor) -> Tensor:
        """
        参数:
            c: 基础特征 [B, C, H, W]
            att: 注意力特征 [B, C, H, W]
            query: 通道权重向量 [B, 512] (来自CLIP)
        返回:
            加权后的特征 [B, C, H, W]
        """
        # 1. 特征融合
        att = c * att + c
        att = self.conv2d(att)

        # 2. 通道注意力计算
        wei = self.avg_pool(att)  # [B, C, 1, 1]
        wei = wei.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        wei = self.conv1d(wei)  # [B, 1, C]

        # 3. 使用query进行通道加权
        # 确保query形状正确
        if query.shape[1] != 512:
            raise ValueError(f"Expected query shape [B, 512], got {query.shape}")

        query = self.u_proj(query).unsqueeze(1)  # [B, 1, C]
        wei = wei * query  # 通道维度加权 [B, 1, C]

        # 4. 激活和形状调整
        wei = self.sigmoid(wei).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]

        # 5. 特征加权和归一化
        x = self.norm(c + att * wei)
        return x

class LFE_Module(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]

        self.mlp = nn.Sequential(*mlp_layer)
        self.LFEA = LFEA(dim, norm_layer, act_layer)

        if stage == 0:
            self.gaussian_0 = Gaussian(dim, 5, 0.2, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x: Tensor,query) -> Tensor:
        # show_feature(x)
        if self.stage == 0:
            att = self.gaussian_0(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att,query)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):
        super().__init__()

        self.blocks = nn.ModuleList([  # Changed from Sequential to ModuleList
            LFE_Module(
                dim=dim,
                stage=stage,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])

    def forward(self, x: Tensor, query: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, query)  # Explicitly pass query to each block
        return x



class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super(LoGFilter, self).__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        """创建高斯-拉普拉斯核"""
        # 初始化二维坐标
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        # 计算高斯-拉普拉斯核
        kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (2 * math.pi * sigma ** 4) * torch.exp(
            -(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        # 归一化
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                             groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]

    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x


class Stem(nn.Module):

    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)  # stem_dim / 2
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            build_norm_layer(norm_layer, out_c12)[1])
        # 定义LoG滤波器
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, norm_layer, act_layer)
        # gaussian
        self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.drfd = DRFD(out_c12, norm_layer, act_layer)

    def forward(self, x):
        x = self.LoG(x)
        # original size to 2x downsampling layer
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)

        return x  # x = [B, C, H/4, W/4]

# class LoGFilter(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
#         super(LoGFilter, self).__init__()
#         # # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
#         # self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
#         """创建高斯-拉普拉斯核"""
#         # 初始化二维坐标
#         ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
#         xx, yy = torch.meshgrid(ax, ax)
#         # 计算高斯-拉普拉斯核
#         kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (2 * math.pi * sigma ** 4) * torch.exp(
#             -(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
#         # 归一化
#         kernel = kernel - kernel.mean()
#         kernel = kernel / kernel.sum()
#         log_kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
#         self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
#                              groups=out_c, bias=False)
#         self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
#         #self.act = act_layer()
#         #self.norm1 = build_norm_layer(norm_layer, out_c)[1]
#         #self.norm2 = build_norm_layer(norm_layer, out_c)[1]
#
#
#
#     def forward(self, x):
#         # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
#         #x = self.conv_init(x)  # x = [B, C/4, H, W]
#         LoG = self.LoG(x)
#         #LoG_edge = self.act(self.norm1(LoG))
#         #x = self.norm2(x + LoG_edge)
#
#         return LoG
#
#
# class Stem(nn.Module):
#     def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
#         super(Stem, self).__init__()
#
#
#         # 定义卷积层和批归一化层
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=in_chans, out_channels=stem_dim, kernel_size=7, stride=2, padding=3),
#             build_norm_layer(norm_layer, stem_dim)[1],
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_chans, out_channels=stem_dim, kernel_size=3, stride=2,padding=1),
#             build_norm_layer(norm_layer, stem_dim)[1],
#         )
#
#
#         self.LoG = nn.Sequential(
#             LoGFilter(stem_dim, stem_dim, 3, 1.0, norm_layer, act_layer),
#             build_norm_layer(norm_layer, stem_dim)[1],
#         )
#
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(stem_dim, out_channels=stem_dim, kernel_size=3, stride=1, padding=1),
#             build_norm_layer(norm_layer, stem_dim)[1],
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(stem_dim, out_channels=stem_dim, kernel_size=1, stride=1),
#             build_norm_layer(norm_layer, stem_dim)[1],
#         )
#         self.bn = build_norm_layer(norm_layer, stem_dim)[1]
#
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(stem_dim, out_channels=stem_dim, kernel_size=1, stride=1),
#             build_norm_layer(norm_layer, stem_dim)[1],
#         )
#
#
#
#
#     def forward(self, x):
#         # 第一部分
#         conv1_out = self.conv1(x)
#         conv2_out = self.conv2(x)
#         sum1 = conv1_out + conv2_out
#         act1 = F.relu(sum1)
#
#
#
#         # 第二部分
#         conv3_out = self.conv3(act1)
#         conv4_out = self.conv4(act1)
#         conv5_out = self.LoG(act1)
#         conv6_out = self.bn(act1)
#         sum2 = conv3_out + conv4_out+ conv5_out + conv6_out
#         act2 = F.relu(sum2)
#
#         # 第三部分
#         conv7_out = self.conv5(act2)
#         conv8_out = self.bn(act2)
#         sum3 = conv7_out + conv8_out
#         act3 = F.relu(sum3)
#         return act3



class MoeLayer(nn.Module):
    def __init__(self, dim, num_experts, hidden_dim, output_dim, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.output_dim = output_dim
        self.num_experts = num_experts

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_experts)
        )

        # 专家池
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
            # 差异化初始化
            nn.init.xavier_normal_(expert[0].weight, gain=1.0 + 0.1 * i)
            nn.init.normal_(expert[0].bias, std=0.01 * (i + 1))
            nn.init.xavier_uniform_(expert[2].weight, gain=1.0 / (1.0 + 0.1 * i))
            nn.init.zeros_(expert[2].bias)
            self.experts.append(expert)

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        logits = self.gate(x)
        weights = torch.softmax(logits / self.temperature, dim=-1)  # (B, num_experts)

        # Top-k 专家选择
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        print(top_indices[0])

        # 使用稀疏计算，只计算被选中的专家
        out = torch.zeros(x.size(0), self.output_dim, device=x.device)
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        expert_outputs = []

        # 为每个选中的专家计算输出
        for expert_idx in range(self.num_experts):
            # 找出哪些样本选择了当前专家
            mask = (top_indices == expert_idx).any(dim=1)
            if mask.any():
                # 只计算这些样本的专家输出
                expert_out = self.experts[expert_idx](x[mask])
                expert_outputs.append(expert_out)

                # 获取这些样本对应的权重
                expert_pos = (top_indices[mask] == expert_idx).nonzero()[:, 1]
                expert_weight = top_weights[mask].gather(1, expert_pos.unsqueeze(1)).squeeze(1)

                # 加权累加到输出
                out[mask] += expert_out * expert_weight.unsqueeze(1)
                expert_counts[expert_idx] = mask.float().sum()

        # 负载均衡损失（KL散度）
        probs = weights.mean(dim=0)
        balance_loss = (probs * torch.log(probs + 1e-12)).sum() + torch.log(torch.tensor(len(self.experts)))

        # 多样性损失（基于专家输出的余弦相似度）
        if len(expert_outputs) > 1:
            # 将所有专家的输出拼接起来（可能每个专家的样本数不同）
            # 这里我们使用一个近似方法：计算每个batch样本中专家输出的相似度
            expert_outputs_all = []
            for expert_idx in range(self.num_experts):
                mask = (top_indices == expert_idx).any(dim=1)
                if mask.any():
                    expert_out = self.experts[expert_idx](x[mask])
                    expert_outputs_all.append(expert_out[:1])  # 只取每个专家的一个样本来近似计算

            if len(expert_outputs_all) > 1:
                expert_outputs_all = torch.stack(expert_outputs_all, dim=1)  # (1, num_active_experts, output_dim)
                expert_outputs_all = expert_outputs_all / (expert_outputs_all.norm(dim=-1, keepdim=True) + 1e-12)
                sim_matrix = torch.bmm(expert_outputs_all, expert_outputs_all.transpose(1, 2))
                mask = ~torch.eye(sim_matrix.size(1), device=x.device).bool()
                diversity_loss = F.relu(sim_matrix[:, mask]).mean()
            else:
                diversity_loss = torch.tensor(0.0, device=x.device)
        else:
            diversity_loss = torch.tensor(0.0, device=x.device)

        # 总辅助损失
        lambda_diversity = 1.0
        aux_loss = balance_loss + lambda_diversity * diversity_loss

        return out, aux_loss


import torch
import torch.nn as nn


class PositionSpecificAssistantClassifier(nn.Module):
    def __init__(self, dim=512, num_classes=10, top_k=10):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.top_k = top_k  # Number of top positions to select

        # Simplified gating network
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),  # Reduced intermediate dimension
            nn.ReLU(),  # Using faster ReLU
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # Shared feature transformation
        self.shared_transform = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU()
        )

        # Assistant classification heads (parameter sharing)
        self.assistant_heads = nn.ModuleList([
            nn.Linear(dim * 2, num_classes)
            for _ in range(49)
        ])

    def forward(self, x):
        # Input shape: (B, C, 7, 7)
        B, C, H, W = x.shape
        assert H == 7 and W == 7, "Input must be a 7x7 feature map"

        # Flatten the input (B, C, 7, 7) -> (B*49, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * 49, C)

        # Calculate activation scores (B*49,)
        scores = self.gate(x_flat).squeeze(-1)

        # Reshape to (B, 49) and select top-k positions
        scores = scores.reshape(B, 49)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=1)
        #print(topk_indices[0])

        # Create binary mask (only 1 for top-k positions, 0 otherwise)
        mask = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)  # (B, 49)
        mask = mask.reshape(-1)  # Flatten to (B*49,)

        # Shared feature transformation (for all positions)
        transformed_x = self.shared_transform(x_flat)  # (B*49, dim*2)

        # Batch process all assistant heads
        all_logits = torch.stack([head(transformed_x) for head in self.assistant_heads],
                                 dim=1)  # (B*49, 49, num_classes)

        # Select logits from corresponding positions
        position_indices = torch.arange(49).repeat(B).to(x.device)  # (B*49,)
        selected_logits = all_logits[torch.arange(B * 49), position_indices]  # (B*49, num_classes)

        # Apply mask (non-top-k positions are zeroed out)
        masked_logits = selected_logits * mask.unsqueeze(-1)  # (B*49, num_classes)

        # Aggregate results (sum over top-k positions only)
        masked_logits = masked_logits.reshape(B, 49, self.num_classes)
        total_logits = masked_logits.sum(dim=1)  # (B, num_classes)

        # Normalize by top_k (optional)
        total_logits = total_logits / self.top_k

        return total_logits

    def get_activation_map(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * 49, C)
        weights = self.gate(x_flat).squeeze(-1)
        return weights.reshape(B, 7, 7)



def Resnet50():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features  # 获取fc层的输入特征数
    model.fc = torch.nn.Linear(num_ftrs, 8)  # 修改fc层为输出17个特征

    weights = torch.load('model_data/resnet50_stl_material.pth')
    model_state_dict = weights['state_dict']
    model.load_state_dict(model_state_dict)

    return model


class LWEGNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=17,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU,
                 mlp_ratio=2.,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))
        hidden_dim = stem_dim * 8
        num_specific_experts = 8
        top_k = 2


        if stem_dim == 96:
            act_layer = nn.ReLU

        self.Stem = Stem(in_chans=in_chans, stem_dim=stem_dim, act_layer=act_layer, norm_layer=norm_layer)

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(stem_dim * 2 ** i_stage),
                               stage=i_stage,
                               depth=depths[i_stage],
                               mlp_ratio=mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DRFD(dim=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        self.forward = self.forward_det
        # add a norm layer for each output
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.init_cfg = copy.deepcopy(init_cfg)


        self.resnet = Resnet50()
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze all layers
        text_promot = np.load('tensor_data.npy')
        self.register_buffer('text_promot', torch.from_numpy(text_promot))

        # self.shared_experts = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim )
        #     ) for _ in range(num_shared_experts)
        # ])

        # 专有专家（动态路由选择）
        self.moe = MoeLayer(
            dim=hidden_dim,
            num_experts=num_specific_experts,
            hidden_dim=hidden_dim * 4,
            output_dim=hidden_dim,  # 注意：输出维度与共享专家对齐
            top_k=top_k
        )

        # 最终分类层
        self.fc1 = nn.Linear(hidden_dim*2, num_classes)

        self.Ass = PositionSpecificAssistantClassifier(dim=hidden_dim, num_classes=hidden_dim, top_k=28 )
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.norm_3 = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction


        MaterialLabels = self.resnet(x)
        MaterialLabels = torch.sigmoid(MaterialLabels)
        idx = torch.argmax(MaterialLabels, dim=1)


        # Use registered buffer that will automatically move to correct device
        query_promot = self.text_promot[idx].float() # This will be on same device as model



        x = self.Stem(x)


        for stage in self.stages:
            if isinstance(stage, BasicStage):
                x = stage(x, query_promot)
            else:
                x = stage(x)

        # 助理计算
        ass = self.Ass(x)
        #print(ass.shape)

        x_mean = x.mean(dim=[-2, -1])  # 全局平均池化 [B, C]

        # 共享专家计算
        # shared_out = 0
        # for expert in self.shared_experts:
        #     shared_out += expert(x_mean)  # 所有共享专家求和
        # shared_out = shared_out / len(self.shared_experts)  # 平均

        # 专有专家计算
        specific_out, aux_loss = self.moe(x_mean)  # [B, hidden_dim]

        # 合并共享和专有专家

        #combined = shared_out + specific_out + ass  # 可改为拼接或其他操作

        combined = torch.cat([self.norm_1(specific_out),self.norm_3(ass)], dim=1)  # 形状 [15, 512]
        combined = self.activation(combined)  # 如GELU/SiLU等


        # 最终分类
        logits = self.fc1(combined)
        # logits = combined
        return logits
# a = torch.rand(5, 3, 224, 224)
# at = LWEGNet()
# B,c = at(a)
# print(B.shape)

def zhong2023_multilabel(num_classes, **kwargs):
    assert num_classes > 1
    model =LWEGNet(num_classes=17)
    return model


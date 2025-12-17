import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
from .ACRNN import acrnn
from .AASIST import *


def apply_masked_feature(feat, p=0.0, max_width_ratio=0.2, axis="time"):
    """
    Feature-level masking: randomly zero a contiguous block along time or channel.
    feat: torch.Tensor [..., T, C] or [..., C] shaped; expects 3D [B, T, C].
    """
    if p <= 0:
        return feat
    if torch.rand(1).item() > p:
        return feat
    x = feat
    if x.dim() < 3:
        return feat
    B, T, C = x.shape
    if axis == "channel":
        width = max(1, int(C * max_width_ratio))
        start = torch.randint(0, max(1, C - width + 1), (1,)).item()
        mask = torch.ones_like(x)
        mask[:, :, start:start + width] = 0
        return x * mask
    width = max(1, int(T * max_width_ratio))
    start = torch.randint(0, max(1, T - width + 1), (1,)).item()
    mask = torch.ones_like(x)
    mask[:, start:start + width, :] = 0
    return x * mask

class LoRAExpert(nn.Module):
    """A single low-rank adapter expert."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 4):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.B(self.A(x))


class MoELoRALinear(nn.Module):
    """
    Wrap a frozen Linear with a mixture-of-LoRA experts + router.
    Only router + experts are trainable.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int = 3,
        rank: int = 4,
        top_k: int = 2,
        router_noise: float = 0.0,
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.router_noise = router_noise
        self.router = nn.Linear(base_linear.in_features, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [LoRAExpert(base_linear.in_features, base_linear.out_features, rank=rank) for _ in range(num_experts)]
        )
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., in_dim]
        base_out = F.linear(x, self.base.weight, self.base.bias)
        router_logits = self.router(x)
        if self.router_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise
        gate = torch.softmax(router_logits, dim=-1)
        if self.top_k < self.num_experts:
            topk = torch.topk(gate, self.top_k, dim=-1)
            gate = torch.zeros_like(gate).scatter(-1, topk.indices, topk.values)
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-6)
        lora_out = 0
        for i, expert in enumerate(self.experts):
            lora_out = lora_out + gate[..., i].unsqueeze(-1) * expert(x)
        return base_out + lora_out

    @property
    def weight(self):
        # expose frozen weight for downstream modules that access .weight
        return self.base.weight

    @property
    def bias(self):
        # expose frozen bias for downstream modules that access .bias
        return self.base.bias


class ASR_model(nn.Module):
    def __init__(
        self,
        moe_lora_enable: bool = False,
        moe_lora_experts: int = 3,
        moe_lora_rank: int = 4,
        moe_lora_topk: int = 2,
        moe_lora_router_noise: float = 0.0,
    ):
        super(ASR_model, self).__init__()
        cp_path = os.path.join('./pretrained_models/xlsr2_300m.pt')   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].cuda()
        if moe_lora_enable:
            self._inject_moe_lora(
                num_experts=moe_lora_experts,
                rank=moe_lora_rank,
                top_k=moe_lora_topk,
                router_noise=moe_lora_router_noise,
            )
        self.linear = nn.Linear(1024, 128)
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)

    def _inject_moe_lora(self, num_experts: int, rank: int, top_k: int, router_noise: float):
        """
        Replace q/k/v/out projections in each transformer layer with MoE-LoRA wrapped versions.
        """
        device = next(self.model.parameters()).device
        for layer in self.model.encoder.layers:
            attn = layer.self_attn
            attn.q_proj = MoELoRALinear(attn.q_proj, num_experts=num_experts, rank=rank, top_k=top_k, router_noise=router_noise).to(device)
            attn.k_proj = MoELoRALinear(attn.k_proj, num_experts=num_experts, rank=rank, top_k=top_k, router_noise=router_noise).to(device)
            attn.v_proj = MoELoRALinear(attn.v_proj, num_experts=num_experts, rank=rank, top_k=top_k, router_noise=router_noise).to(device)
            attn.out_proj = MoELoRALinear(attn.out_proj, num_experts=num_experts, rank=rank, top_k=top_k, router_noise=router_noise).to(device)

    def forward(self, x):
        emb = self.model(x, mask=False, features_only=True)['x']
        emb = self.linear(emb) 
        emb = F.max_pool2d(emb, (4,2)) 
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb

    
class SER_model(nn.Module):
    def __init__(self):
        super(SER_model, self).__init__()
        cp_path = os.path.join('./pretrained_models/ser_acrnn.pth')   # Change the pre-trained SER model path. 
        model=acrnn().cuda()
        model.load_state_dict(torch.load(cp_path))
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)
        self.model = model

    def forward(self, x):
        emb = self.model(x)
        emb = F.max_pool2d(emb, (3, 4)) 
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb

    
class base_encoder(nn.Module):
    def __init__(self):
        super(base_encoder, self).__init__()
        filts= [[1, 32], [32, 32], [32, 64], [64, 64]]
        self.conv_time=CONV(out_channels=70,
                              kernel_size=128,
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[0], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[1])),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3]))) 

    def forward(self, x, Freq_aug):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        emb = self.encoder(x) 
        out, _ = torch.max(torch.abs(emb), dim=2) 
        out = out.transpose(1, 2) 
        return out


class HGFM(nn.Module):
    def __init__(self):
        super(HGFM, self).__init__()
        self.HtrgGAT_layer1 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.HtrgGAT_layer2 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.stack_node = nn.Parameter(torch.randn(1, 1, 64))

    def forward(self, x1, x2):
        stack_node = self.stack_node.expand(x1.size(0), -1, -1)
        x1, x2, stack_node,_ = self.HtrgGAT_layer1(x1, x2, master=stack_node)
        x1_aug, x2_aug, stack_node2,attmap = self.HtrgGAT_layer2(x1, x2, master=stack_node)
        x1 = x1 + x1_aug 
        x2 = x2 + x2_aug
        stack_node = stack_node + stack_node2 
        x1 = self.drop_way(x1)
        x2 = self.drop_way(x2)
        stack_node = self.drop_way(stack_node)
        return x1+x2, stack_node, attmap


class GRS(nn.Module):
    def __init__(self):
        super(GRS, self).__init__()
        self.pool1 = GraphPool(0.5, 64, 0.3)
        self.pool2 = GraphPool(0.5, 64, 0.3)
    def forward(self, x_list):
        pool_list=[]
        for i in x_list:
            pool_list.append(self.pool2(self.pool1(i)))
        pool_cat=torch.cat(pool_list, dim=1)
        pool_max, _=torch.max(torch.abs(pool_cat),dim=1)
        pool_avg=torch.mean(pool_cat,dim=1)
        return torch.cat([pool_max,pool_avg], dim=1)


class Module(nn.Module):
    def __init__(
        self,
        feature_mask_p: float = 0.0,
        feature_mask_axis: str = "time",
        feature_mask_max_ratio: float = 0.2,
        moe_lora_enable: bool = False,
        moe_lora_experts: int = 3,
        moe_lora_rank: int = 4,
        moe_lora_topk: int = 2,
        moe_lora_router_noise: float = 0.0,
    ):
        super(Module, self).__init__()
        """multi-view feature extractor"""
        self.text_view_extract=ASR_model(
            moe_lora_enable=moe_lora_enable,
            moe_lora_experts=moe_lora_experts,
            moe_lora_rank=moe_lora_rank,
            moe_lora_topk=moe_lora_topk,
            moe_lora_router_noise=moe_lora_router_noise,
        )
        self.emo_view_extract=SER_model()
        self.audio_view_extract=base_encoder()
        self.feature_mask_p = feature_mask_p
        self.feature_mask_axis = feature_mask_axis
        self.feature_mask_max_ratio = feature_mask_max_ratio
        """IGAM"""
        self.GAT_text = GraphAttentionLayer(64, 64, temperature=2.0)
        self.pool_text = GraphPool(0.5, 64, 0.3)
        self.GAT_emo = GraphAttentionLayer(64, 64, temperature=2.0)
        self.pool_emo = GraphPool(0.5, 64, 0.3)
        self.GAT_audio= GraphAttentionLayer(64, 64, temperature=2.0)
        self.pool_audio = GraphPool(0.88, 64, 0.3)
        """HGFM"""
        self.Core_AE = HGFM()
        self.Core_AT = HGFM()
        self.Core_ET = HGFM()
        self.Core_AET = HGFM()
        """GRS"""
        self.GRS_group1=GRS()
        self.GRS_group2=GRS()
        self.GRS_group3=GRS()
        self.drop = nn.Dropout(0.5, inplace=True)
        self.out_layer = nn.Linear(384, 64)
        self.out_layer2 = nn.Linear(64, 2)
        
    def forward(self, inputs,inputs2, Freq_aug):
        x=inputs
        x2=inputs2
        """multi-view features"""
        audio_view=self.audio_view_extract(x, Freq_aug=Freq_aug) 
        emo_view=self.emo_view_extract(x2)
        text_view=self.text_view_extract(x)
        if self.training and self.feature_mask_p>0:
            audio_view = apply_masked_feature(audio_view, p=self.feature_mask_p, max_width_ratio=self.feature_mask_max_ratio, axis=self.feature_mask_axis)
            emo_view = apply_masked_feature(emo_view, p=self.feature_mask_p, max_width_ratio=self.feature_mask_max_ratio, axis=self.feature_mask_axis)
            text_view = apply_masked_feature(text_view, p=self.feature_mask_p, max_width_ratio=self.feature_mask_max_ratio, axis=self.feature_mask_axis)
        """ Intra-view graph attention module"""
        emo_gat = self.GAT_emo(emo_view) 
        audio_gat = self.GAT_audio(audio_view) 
        text_gat = self.GAT_text(text_view) 
        emo_gat = self.pool_emo(emo_gat) 
        audio_gat = self.pool_audio(audio_gat) 
        text_gat = self.pool_text(text_gat)
        """ Heterogeneous graph fusion module"""
        AE_HG, AE_SN,attmap_AE = self.Core_AE(audio_gat, emo_gat) # A-E
        AT_HG, AT_SN,attmap_AT = self.Core_AT(audio_gat, text_gat) # A-T
        ET_HG, ET_SN,attmap_ET = self.Core_ET(emo_gat, text_gat) # E-T
        AET_HG, AET_SN,attmap_AET = self.Core_AET(AE_HG, ET_HG) # A-E-T
        """Group-based Readout Scheme"""
        GAT_Group=[audio_gat,emo_gat,text_gat]
        HGAT_Group=[AE_HG,AT_HG,ET_HG,AET_HG]
        SN_Group=[AE_SN,AT_SN,ET_SN,AET_SN]
        out1=self.GRS_group1(GAT_Group)
        out2=self.GRS_group2(HGAT_Group)
        out3=self.GRS_group3(SN_Group)
        """output"""
        last_hidden = torch.cat([out1,out2,out3], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        output = self.out_layer2(output)
        return output

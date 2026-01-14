import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple, Mlp, DropPath

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)

### --- NEW: Temporal Enhancer Module --- ###
class TemporalEnhancer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # 1. Feature Enhancement (Cross-Attention)
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        
        # 2. 独立的 FFN (可选，但推荐，为了让时序特征更成熟)
        # 如果觉得参数量太大，可以把 mlp_ratio 设小一点，比如 2.0 或 1.0
        self.norm_enhance = norm_layer(dim)
        self.enhance_ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        
        # 3. Gate MLP (计算融合权重)
        gate_hidden_dim = int(dim * 0.25)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, dim),
            nn.Sigmoid()
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.trace_data = {}

        self._init_weights()

    def _init_weights(self):
        # Zero-init Cross-Attention
        nn.init.constant_(self.cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0)
        
        # Zero-init Enhance FFN
        if hasattr(self.enhance_ffn, 'fc2'):
            nn.init.constant_(self.enhance_ffn.fc2.weight, 0)
            nn.init.constant_(self.enhance_ffn.fc2.bias, 0)

        # Zero-init Gate
        nn.init.constant_(self.gate_mlp[2].weight, 0)
        nn.init.constant_(self.gate_mlp[2].bias, 0)

    def forward(self, x, prev_result, global_index_t, confidence_score):
        # x: [B, N, C]
        lens_z_current = global_index_t.shape[1]
        t_tokens = x[:, :lens_z_current]
        s_tokens = x[:, lens_z_current:]

        # --- Branch 1: 时序增强特征提取 ---
        # 1. Cross Attention
        q = self.norm_q(t_tokens)
        kv = self.norm_kv(prev_result)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        
        # 2. 独立的 FFN 处理增强特征 (Post-Attn Processing)
        # 这一步让特征在融合前完成非线性变换
        feat_enhance = attn_out + self.drop_path(self.enhance_ffn(self.norm_enhance(attn_out)))
        
        # --- Branch 2: 计算 Gate ---
        alpha_internal = self.gate_mlp(feat_enhance) # 根据处理好的特征计算置信度
        alpha_external = confidence_score.unsqueeze(-1).unsqueeze(-1)
        effective_gate = alpha_internal * alpha_external

        if not self.training:
            # 记录一些中间数据，便于分析
            # alpha_internal 是 [B, N, C]，取均值变成标量
            self.trace_data['alpha_internal_mean'] = alpha_internal.mean().item()
            
            # alpha_external 是 [B, 1, 1]，取均值（或直接取值）变成标量
            self.trace_data['alpha_external'] = alpha_external.mean().item()
            
            # effective_gate 是 [B, N, C]，取均值变成标量
            self.trace_data['effective_gate_mean'] = effective_gate.mean().item()

        # --- Fusion: Post-Process Fusion ---
        # 将增强特征融合回原始的主干 token
        t_enhanced = t_tokens + effective_gate * feat_enhance

        # 如果这个模块仅仅是做增强（不替代原本的 Block），到这里就结束了
        # 返回融合后的 t 和原始的 s
        return torch.cat([t_enhanced, s_tokens], dim=1)
### --- END NEW --- ###

### --- NEW: Positional Encoding Generator (CPE) --- ###
class PEG(nn.Module):
    def __init__(self, dim=256, k=3):
        super(PEG, self).__init__()
        # Depth-wise Convolution: groups=dim
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)

    def forward(self, x, H, W):
        # x: [B, N, C]
        B, N, C = x.shape
        # Reshape to [B, C, H, W] for Conv2d
        feat = x.transpose(1, 2).view(B, C, H, W)
        # Apply Conv and add residual connection
        # PEG(x) + x
        cnn_feat = self.proj(feat) + feat
        # Flatten back to [B, N, C]
        x = cnn_feat.flatten(2).transpose(1, 2)
        return x
### --- END NEW --- ###


class VisionTransformerCEF(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None,
                 store_feature_loc=None, temporal_enhance_loc=None): ### --- MODIFIED --- ###
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            store_feature_loc (list): (NEW) list of block indices to store features from.
            temporal_enhance_loc (list): (NEW) list of block indices to inject temporal features.
            freeze_backbone (bool): (NEW) Whether to freeze the ViT backbone parameters.
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)

        # import ipdb; ipdb.set_trace();
        self.patch_size = patch_size
        self.in_chans = in_chans

        # import ipdb; ipdb.set_trace()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        ### --- NEW: Initialize PEG (CPE) --- ###
        # self.peg = PEG(dim=embed_dim)
        # Critical: Zero-initialization to keep pre-trained weights valid at start
        # nn.init.zeros_(self.peg.proj.weight)
        # nn.init.zeros_(self.peg.proj.bias)
        ### --- END NEW --- ###

        ### --- NEW --- ###
        self.store_feature_loc = store_feature_loc
        self.temporal_enhance_loc = temporal_enhance_loc
        # Initialize your enhancement modules (Cross-Attention)
        if self.temporal_enhance_loc is not None:
            self.temporal_enhancers = nn.ModuleDict()
            for i in self.temporal_enhance_loc:
                self.temporal_enhancers[str(i)] = TemporalEnhancer(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], # Use same dpr
                    norm_layer=norm_layer, act_layer=act_layer
                )
        ### --- END NEW --- ###

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, 
                         temporal_data=None
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        C_embed = x.shape[-1]
        H_x, W_x = x.shape[1]**.5, x.shape[1]**.5
        H_z, W_z = z.shape[1]**.5, z.shape[1]**.5
        H_x, W_x = int(H_x), int(W_x)
        H_z, W_z = int(H_z), int(W_z)
        pos_embed_x_2d = self.pos_embed_x.transpose(1, 2).reshape(1, C_embed, H_x, W_x)
        self.pos_embed_x_2d = pos_embed_x_2d.expand(B, -1, -1, -1).contiguous()

        ### --- NEW: Apply PEG (CPE) --- ###
        # 1. 计算 Feature Map 的尺寸
        # x shape: [B, N_x, C], z shape: [B, N_z, C]
        # H_feat = H_img // patch_size
        
        # 转换为 int
        H_x, W_x = int(H_x), int(W_x)
        H_z, W_z = int(H_z), int(W_z)

        # 2. 分别应用 PEG
        # 此时 z 和 x 拥有独立的 2D 空间结构，卷积效果最好
        # z = self.peg(z, H_z, W_z)
        # x = self.peg(x, H_x, W_x)
        ### --- END NEW --- ###

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        
        ### --- NEW --- ###
        feature_to_store = None

        if temporal_data is not None:
            prev_features = temporal_data.get('features', None)
            prev_confidence = temporal_data.get('confidence', None)
        else:
            prev_features = None
            prev_confidence = None
        ### --- END NEW --- ###

        for i, blk in enumerate(self.blocks):
            ### --- NEW: Placeholder for Temporal Enhancement --- ###
            if self.temporal_enhance_loc is not None and i in self.temporal_enhance_loc and prev_features is not None:
                # Use a default confidence of 1.0 if not provided
                if prev_confidence is None:
                    prev_confidence = torch.ones(B, device=x.device)
                
                enhancer_module = self.temporal_enhancers[str(i)]
                x = enhancer_module(x, prev_features, global_index_t, prev_confidence)
            ### --- END NEW --- ###

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
            
            ### --- NEW: Store features for next frame --- ###
            if self.store_feature_loc is not None and i in self.store_feature_loc:
                # Get current search tokens (pruned)
                lens_z_current = global_index_t.shape[1]
                lens_x_current = global_index_s.shape[1]
                s_tokens_pruned = x[:, lens_z_current:] # Shape: [B, lens_x_current, C]

                # import ipdb; ipdb.set_trace();

                C = s_tokens_pruned.shape[2]
                
                # Un-prune them to get the full token list
                s_tokens_unpruned = s_tokens_pruned
                
                # Collect all indices removed so far
                current_removed_indexes = [idx for idx in removed_indexes_s if idx is not None]

                if current_removed_indexes:
                    removed_indexes_cat = torch.cat(current_removed_indexes, dim=1)
                    pruned_lens_x = lens_x - lens_x_current
                    
                    pad_x = torch.zeros([B, pruned_lens_x, C], device=s_tokens_pruned.device)
                    s_tokens_with_pad = torch.cat([s_tokens_pruned, pad_x], dim=1)
                    
                    index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                    
                    s_tokens_unpruned = torch.zeros_like(s_tokens_with_pad).scatter_(
                        dim=1, 
                        index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), 
                        src=s_tokens_with_pad
                    )
                
                # Reshape to feature map [B, C, H, W]
                H_feat, W_feat = self.search_grid_size
                # import ipdb; ipdb.set_trace();
                feature_to_store = s_tokens_unpruned.transpose(1, 2).reshape(B, C, H_feat, W_feat).detach()
            ### --- END NEW --- ###


        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "feature_to_store": feature_to_store,      ### --- NEW --- ###
            "observation_data": self.temporal_enhancers[str(self.temporal_enhance_loc[-1])].trace_data if self.temporal_enhance_loc is not None else None ### --- NEW --- ###
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,
                temporal_data=None): ### --- MODIFIED --- ###

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            temporal_data=temporal_data) ### --- MODIFIED --- ###

        return x, aux_dict

    # --- 新增的辅助函数 ---
    def display_trainable_info(self):
        """
        统计并打印模型的可训练参数与冻结参数概览
        """
        total_params = 0
        trainable_params = 0
        trainable_names = set()
        
        print("\n" + "=" * 60)
        print(f"{'Model Parameter Configuration':^60}")
        print("=" * 60)

        for name, param in self.named_parameters():
            num = param.numel()
            total_params += num
            
            if param.requires_grad:
                trainable_params += num
                # 为了防止打印列表过长，进行简单的层级聚合
                # 例如 blocks.0.norm1.weight -> blocks.0
                # temporal_enhancers.0.cross_attn... -> temporal_enhancers.0
                parts = name.split('.')
                if len(parts) >= 2:
                    group_name = f"{parts[0]}.{parts[1]}"
                else:
                    group_name = parts[0]
                trainable_names.add(group_name)

        frozen_params = total_params - trainable_params
        trainable_ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0

        # 打印统计数据
        print(f"Total Parameters:     {total_params / 1e6:.2f} M")
        print(f"Frozen Parameters:    {frozen_params / 1e6:.2f} M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M ({trainable_ratio:.2f}%)")
        
        # 打印可训练模块概览
        print("-" * 60)
        print("Trainable Modules (Grouped):")
        # 排序让输出更好看
        for name in sorted(list(trainable_names)):
            print(f"  -> {name}.*")
            
        print("=" * 60 + "\n")


def _create_vision_transformer_cef(pretrained=False, **kwargs):
    model = VisionTransformerCEF(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model

def vit_tiny_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer_cef(pretrained=pretrained, **model_kwargs)
    return model
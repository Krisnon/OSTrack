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
    """
    Temporal Enhancement block using Cross-Attention and Gating.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # 1. Feature Enhancement Module (Cross-Attention)
        self.norm_q = norm_layer(dim)  # Norm for Query (Current Template)
        self.norm_kv = norm_layer(dim) # Norm for Key/Value (Previous Result)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        
        # 2. Confidence Assessment Module (Gating)
        # This small MLP learns an "internal" confidence score
        gate_hidden_dim = int(dim * 0.25)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, dim),
            nn.Sigmoid()
        )
        
        # Standard Transformer block components
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_features = int(dim * mlp_ratio)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_features, act_layer=act_layer, drop=drop)

        # zero initialization
        self._init_weights()

    def _init_weights(self):
        """
        Apply Zero-Initialization to ensure the module acts as an Identity function 
        at the start of training.
        """
        # 1. Zero-init Cross-Attention Output Projection
        # 这样 attn_output 和 delta_t 在初始时全为 0
        nn.init.constant_(self.cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0)

        # 2. Zero-init FFN Final Projection
        # 这样 FFN 的输出在初始时全为 0
        # 注意：这里假设使用的是 timm.models.layers.Mlp，最后一层通常是 fc2
        if hasattr(self.ffn, 'fc2'):
            nn.init.constant_(self.ffn.fc2.weight, 0)
            nn.init.constant_(self.ffn.fc2.bias, 0)
        elif hasattr(self.ffn, 'layers') and isinstance(self.ffn.layers[-1], nn.Linear):
             # 如果是自定义的 Sequential Mlp
             nn.init.constant_(self.ffn.layers[-1].weight, 0)
             nn.init.constant_(self.ffn.layers[-1].bias, 0)

        # 3. (可选) Zero-init Gate MLP 的最后一层 Linear
        # 虽然 delta_t 已经是 0 了，Gate 的值在初始时并不重要（0 * 任何数 = 0），
        # 但通常也会初始化 Gate 使其趋向于平滑。
        # 这里我们将 Gate 的最后一层 Linear 初始化为 0，这会导致 Sigmoid 输入为 0，输出为 0.5。
        nn.init.constant_(self.gate_mlp[2].weight, 0)
        nn.init.constant_(self.gate_mlp[2].bias, 0)

    def forward(self, x, prev_result, global_index_t, confidence_score):
        """
        Args:
            x (torch.Tensor): The *entire* token sequence [B, N_t + N_s, C]
            prev_result (torch.Tensor): The RoIAlign'd features from t-1, R_{t-1} [B, K*K, C]
            global_index_t (torch.Tensor): Indices of template tokens [B, N_t]
            confidence_score (torch.Tensor): External confidence score from t-1 [B]
        """
        
        # 1. Split tokens
        lens_z_current = global_index_t.shape[1]
        t_tokens = x[:, :lens_z_current]  # Current Template Tokens [B, N_t, C]
        s_tokens = x[:, lens_z_current:]  # Current Search Tokens [B, N_s, C]
        
        # 2. Feature Enhancement (Cross-Attention)
        # Query = t_tokens, Key/Value = prev_result (R_{t-1})
        q = self.norm_q(t_tokens)
        kv = self.norm_kv(prev_result)
        attn_output, _ = self.cross_attn(query=q, key=kv, value=kv)
        
        # delta_t is the "enhancement vector"
        delta_t = self.drop_path(attn_output)

        # 3. Confidence Assessment & Gating
        # 3a. Internal Confidence: Gate learns "how useful" delta_t is
        alpha_internal = self.gate_mlp(delta_t)
        
        # 3b. External Confidence: Reshape [B] -> [B, 1, 1] for broadcasting
        alpha_external = confidence_score.unsqueeze(-1).unsqueeze(-1)
        
        # 3c. Combined Gate:
        effective_gate = alpha_internal * alpha_external
        
        # 3d. Apply Gated Update:
        # t_tokens + (effective_gate * delta_t)
        t_gated = t_tokens.addcmul(effective_gate, delta_t)

        # 4. Standard FFN block
        t_final = t_gated + self.drop_path(self.ffn(self.norm2(t_gated)))
        
        # 5. Recombine tokens and return
        return torch.cat([t_final, s_tokens], dim=1)
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
                 store_feature_loc=None, temporal_enhance_loc=None,
                 freeze_backbone=False): ### --- MODIFIED --- ###
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

        ### --- NEW: Freeze Backbone Logic --- ###
        if freeze_backbone:
            _logger.info("Freezing backbone parameters (ViT), keeping Temporal Enhancer trainable.")
            for name, param in self.named_parameters():
                # 核心逻辑：如果参数名中包含 'temporal_enhancers'，则保留梯度（可训练）
                # 否则（属于 patch_embed, blocks, pos_embed, cls_token 等），冻结梯度
                if 'temporal_enhancers' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        ### --- END NEW --- ###

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

        ### --- NEW: Apply PEG (CPE) --- ###
        # 1. 计算 Feature Map 的尺寸
        # x shape: [B, N_x, C], z shape: [B, N_z, C]
        # H_feat = H_img // patch_size
        H_x, W_x = x.shape[1]**.5, x.shape[1]**.5
        H_z, W_z = z.shape[1]**.5, z.shape[1]**.5
        
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
            "feature_to_store": feature_to_store      ### --- NEW --- ###
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,
                temporal_data=None): ### --- MODIFIED --- ###

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            temporal_data=temporal_data) ### --- MODIFIED --- ###

        return x, aux_dict


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
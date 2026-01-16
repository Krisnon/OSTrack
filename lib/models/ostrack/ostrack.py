"""
Basic OSTrack model.
"""
import math
import os
import sys
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from torchvision.ops import roi_align ### NEW ###
from torchinfo import summary

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce_fusion import vit_tiny_patch16_224_ce
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy ### MODIFIED ###


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 backbone_stride=16, temporal_roi_size=5, 
                 use_teacher_forcing=False, roi_jitter_scale=0.0): ### MODIFIED ###
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            backbone_stride (int): (NEW) The stride of the backbone (e.g., 16 for ViT-B/16).
            temporal_roi_size (int): (NEW) The fixed output size (K) for RoIAlign (e.g., 5 for 5x5).
            use_teacher_forcing (bool): (NEW) Whether to use GT boxes for RoIAlign during training.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        ### --- NEW: Temporal Module Parameters --- ###
        self.backbone_stride = backbone_stride
        self.temporal_roi_size = temporal_roi_size
        self.use_teacher_forcing = use_teacher_forcing
        ### --- END NEW --- ###

        ### --- NEW：Jitter Parameters --- ###
        self.roi_jitter_scale = roi_jitter_scale
        ### --- END NEW --- ###

        ### --- NEW: Content-Aware PE Eraser --- ###
        embed_dim = self.backbone.embed_dim

        self.pe_eraser = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), # feature map & initial pe
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        nn.init.constant_(self.pe_eraser[-1].weight, 0)
        nn.init.constant_(self.pe_eraser[-1].bias, 0)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                gt_bboxes: torch.Tensor = None, ### NEW: GT BBoxes for Teacher Forcing
                temporal_data=None, ### NEW: Receive state from t-1
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        
        ### --- MODIFIED: Pass temporal_data to backbone --- ###
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    temporal_data=temporal_data, ### NEW ###
                                    )
        ### --- END MODIFIED --- ###

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        ### --- NEW: Temporal State Preparation for t+1 --- ###
        
        # 1. Get the feature map F_t that we stored in the backbone
        F_t = aux_dict.get("feature_to_store")
        next_temporal_data = None

        if F_t is not None:
            B, C, H_feat, W_feat = F_t.shape
            
            # 2. Decide which BBox to use for RoIAlign
            # We use unnormalized (image) coordinates for RoIAlign
            img_sz = search.shape[-1] # Assumes square search region, e.g., 256
            
            if self.use_teacher_forcing and gt_bboxes is not None:
                roi_bboxes_cxcywh = gt_bboxes * img_sz # Un-normalize from 0-1
            else:
                # Use the model's own prediction (self-regression)
                roi_bboxes_cxcywh = out['pred_boxes'].squeeze(1) * img_sz # Un-normalize from 0-1

            ### --- NEW: Apply Random Jitter (Robustness) --- ###
            # Only apply jitter during training to simulate tracking errors
            if self.training and self.roi_jitter_scale > 0:
                # roi_bboxes_cxcywh: [B, 4] -> cx, cy, w, h
                # Generate random noise in range [-1, 1]
                noise = (torch.rand_like(roi_bboxes_cxcywh) - 0.5) * 2.0 
                
                # Jitter Position (cx, cy): shift relative to box size
                # e.g. cx_new = cx + noise * scale * w
                roi_bboxes_cxcywh[:, 0] += noise[:, 0] * self.roi_jitter_scale * roi_bboxes_cxcywh[:, 2]
                roi_bboxes_cxcywh[:, 1] += noise[:, 1] * self.roi_jitter_scale * roi_bboxes_cxcywh[:, 3]
                
                # Jitter Size (w, h): scale box
                # e.g. w_new = w * (1 + noise * scale)
                roi_bboxes_cxcywh[:, 2] *= (1.0 + noise[:, 2] * self.roi_jitter_scale)
                roi_bboxes_cxcywh[:, 3] *= (1.0 + noise[:, 3] * self.roi_jitter_scale)
            ### --- END NEW --- ###
            
            # 3. Get confidence score
            # We take the max value from the score map as the confidence
            if self.head_type == "CENTER":
                score_map = out['score_map'] # [B, 1, H_feat, W_feat]
            elif self.head_type == "CORNER":
                score_map = out['score_map'] # [B, 1, H_feat, W_feat]
            
            # [B, 1, H_feat, W_feat] -> [B, H_feat*W_feat] -> [B]
            confidence = score_map.flatten(1).max(dim=1)[0].detach()

            # 4. Calculate RoI-Area-Based Reliability (Your Idea)
            roi_w = roi_bboxes_cxcywh[:, 2] # width
            roi_h = roi_bboxes_cxcywh[:, 3] # height
            
            input_area_feat = (roi_w / self.backbone_stride) * (roi_h / self.backbone_stride)
            output_area = self.temporal_roi_size * self.temporal_roi_size
            
            # Reliability score is the ratio, capped at 1.0
            reliability = (input_area_feat / output_area).clamp(max=1.0).detach()
            
            # 5. Perform RoIAlign
            # Convert [B, 4] (cxcywh) to [B, 4] (xyxy)
            roi_bboxes_xyxy = box_cxcywh_to_xyxy(roi_bboxes_cxcywh)
            
            # Prepare boxes for roi_align: needs [K, 5] (batch_idx, x1, y1, x2, y2)
            batch_indices = torch.arange(B, device=F_t.device, dtype=torch.float32).unsqueeze(1)
            boxes_for_roi = torch.cat([batch_indices, roi_bboxes_xyxy], dim=1)
            
            # R_t is the result for the *next* frame
            # We use spatial_scale=1.0 because F_t is already scaled, 
            # and our bboxes are *not* scaled (they are in original image coords).
            # Wait, no, spatial_scale maps box coords to input coords.
            # Our boxes are on 256px scale, F_t is on 16px scale.
            # So spatial_scale = 16 / 256 = 1.0 / self.backbone_stride
            R_t = roi_align(
                F_t, 
                boxes_for_roi, 
                output_size=(self.temporal_roi_size, self.temporal_roi_size),
                spatial_scale = 1.0 / self.backbone_stride,
                aligned=True
            )

            # PE_roi is the positional embeddings for the RoI region
            PE_roi = roi_align(
                self.backbone.pos_embed_x_2d,
                boxes_for_roi,
                output_size=(self.temporal_roi_size, self.temporal_roi_size),
                spatial_scale = 1.0 / self.backbone_stride,
                aligned=True
            )

            R_t_tokens = R_t.flatten(2).transpose(1, 2)  # [B, K*K, C]
            PE_roi_tokens = PE_roi.flatten(2).transpose(1, 2)  # [B, K*K, C]

            # Apply PE Eraser
            R_t_erased = self.pe_eraser(torch.cat([R_t_tokens, PE_roi_tokens], dim=-1))

            # Apply Template PE
            PE_template_size = int(self.backbone.pos_embed_z.shape[1] ** 0.5)
            
            if(PE_template_size != self.temporal_roi_size):
                C = self.backbone.pos_embed_z.shape[-1]
                PE_template_2d = self.backbone.pos_embed_z.transpose(1, 2).reshape(1, C, PE_template_size, PE_template_size)
                PE_template_resized = nn.functional.interpolate(
                    PE_template_2d,
                    size=(self.temporal_roi_size, self.temporal_roi_size),
                    mode='bilinear',
                    align_corners=False
                )
                PE_template = PE_template_resized.flatten(2).transpose(1, 2)  # [1, K*K, C]
            else:
                PE_template = self.backbone.pos_embed_z  # [1, K*K, C]

            final_temporal_tokens = R_t_erased + PE_template

            # 6. Package the state for the next time step
            next_temporal_data = {
                'features': final_temporal_tokens.detach(),      # [B, C, K, K]
                'confidence': confidence,      # [B]
                'reliability': reliability     # [B]
            }

        ### --- END NEW --- ###

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['next_temporal_data'] = next_temporal_data ### NEW: Return the state for t+1
        out['observation_data'] = aux_dict.get("observation_data", None) ### NEW: Return observation data
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    ### --- NEW: Get backbone stride from config --- ###
    backbone_stride = cfg.MODEL.BACKBONE.STRIDE
    
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           ### --- NEW: Pass temporal params to backbone --- ###
                                           store_feature_loc=cfg.MODEL.TEMPORAL.STORE_LOC,
                                           temporal_enhance_loc=cfg.MODEL.TEMPORAL.ENHANCE_LOC,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            ### --- NEW: Pass temporal params to backbone --- ###
                                            store_feature_loc=cfg.MODEL.TEMPORAL.STORE_LOC,
                                            temporal_enhance_loc=cfg.MODEL.TEMPORAL.ENHANCE_LOC,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224_ce':
        backbone = vit_tiny_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           ### --- NEW: Pass temporal params to backbone --- ###
                                        #    img_size=cfg.DATA.SEARCH.SIZE,
                                           store_feature_loc=cfg.MODEL.TEMPORAL.STORE_LOC,
                                           temporal_enhance_loc=cfg.MODEL.TEMPORAL.ENHANCE_LOC
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    ### --- NEW: Get jitter scale from CFG or Default --- ###
    # 尝试从 CFG 读取，如果没有则默认为 0.0 (关闭) 或 0.1 (开启)
    roi_jitter_scale = getattr(cfg.TRAIN, "ROI_JITTER_SCALE", 0.0)
    ### --- END NEW --- ###

    ### --- MODIFIED: Pass temporal params to OSTrack --- ###
    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        backbone_stride=backbone_stride,
        temporal_roi_size=cfg.MODEL.TEMPORAL.ROI_SIZE,
        use_teacher_forcing=cfg.TRAIN.USE_TEACHER_FORCING,
        roi_jitter_scale=roi_jitter_scale
    )
    ### --- END MODIFIED --- ###

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' '..._') # (Path hidden for brevity)
        ### --- NEW: Print missing keys to verify --- ###
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    # NEW : Freeze some parameters
    print("\n" + "=" * 40)
    print("   INFO: Freezing All Pretrained Parameters")
    print("=" * 40 + "\n")
    
    for name, param in model.named_parameters():
        if 'temporal_enhancers' in name:
            param.requires_grad = True
        elif 'box_head' in name:
            param.requires_grad = True
        elif 'pe_eraser' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    summary(model, [(1, 3, 128, 128), (1, 3, 256, 256)], depth=3)

    return model
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from collections import OrderedDict ### NEW ###


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        
        ### --- MODIFIED: Implement temporal unrolling loop --- ###
        is_training = self.net.training
        
        # 1. Get static template T_0
        # (N_t, B, C, H, W) --> (B, C, H, W)
        # We assume N_t = 1 (static template)
        assert len(data['template_images']) == 1, "Only support 1 template image"
        template_t0 = data['template_images'][0].view(-1, *data['template_images'].shape[2:])

        # 2. Get search image sequence
        # (N_s, B, C, H, W)
        search_images = data['search_images']
        gt_bboxes = data['search_anno'] # (N_s, B, 4)
        
        num_sequence = search_images.shape[0] # N_s
        
        # 3. Get static CE parameters
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_t0.shape[0], template_t0.device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1, # This might need adjustment
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # 4. Initialize loop state
        temporal_data = None
        total_loss = 0.0
        all_status_dicts = []
        if is_training:
            self.optimizer.zero_grad()

        # 5. Unroll the sequence
        for t in range(num_sequence):
            # Get data for time step t
            search_t = search_images[t].view(-1, *search_images.shape[2:]) # (B, C, H, W)
            gt_bbox_t = gt_bboxes[t] # (B, 4)
            
            # Forward pass for one time step
            out_dict = self.forward_pass(template_t0, search_t, gt_bbox_t, 
                                         temporal_data, box_mask_z, ce_keep_rate)
            
            # Compute losses for one time step
            loss_t, status_t = self.compute_losses(out_dict, gt_bbox_t)

            mean_loss_t = loss_t / num_sequence
            if is_training:
                mean_loss_t.backward()
            
            # Accumulate loss and update state
            total_loss += mean_loss_t.item()
            all_status_dicts.append(status_t)
            
            if out_dict.get('next_temporal_data', None) is not None:
                temporal_data = {
                    'features': out_dict['next_temporal_data']['features'].detach(),
                    'confidence': out_dict['next_temporal_data']['confidence'].detach(),
                    'reliability': out_dict['next_temporal_data']['reliability'].detach(),
                }
            else:
                temporal_data = None

        # 6. Compute average loss and status
        mean_loss = total_loss
        
        # Aggregate status dicts
        mean_status = all_status_dicts[0]
        if num_sequence > 1:
            for key in mean_status.keys():
                # Sum up all values for this key
                sum_val = sum(s[key] for s in all_status_dicts)
                mean_status[key] = sum_val / num_sequence

        return mean_loss, mean_status
        ### --- END MODIFIED --- ###

    def forward_pass(self, template, search, gt_bbox, temporal_data, box_mask_z, ce_keep_rate):
        """
        Modified to run one time step.
        """
        
        ### --- MODIFIED --- ###
        if isinstance(template, list):
             # This actor was built assuming a list of templates
             # Our new logic provides a single T_0
             template = template[0] 
        ### --- END MODIFIED --- ###

        out_dict = self.net(template=template,
                            search=search,
                            gt_bboxes=gt_bbox, # Pass GT box for RoIAlign
                            temporal_data=temporal_data, # Pass t-1 state
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        """
        Modified to accept a single GT BBox [B, 4]
        """
        
        # gt gaussian map
        ### --- MODIFIED --- ###
        # gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        # We now receive gt_bbox [B, 4] directly
        
        # generate_heatmap expects a list of (B, 4) tensors
        gt_gaussian_maps = generate_heatmap([gt_bbox], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[0].unsqueeze(1) # [B, 1, H, W]
        ### --- END MODIFIED --- ###


        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = OrderedDict({"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()})
            return loss, status
        else:
            return loss
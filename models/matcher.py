# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1, cost_points: float = 1):
        """Creates the matcher

        Params:
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_points = cost_points
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if "pred_block" in outputs.keys() or "pred_line1" in outputs.keys():
            for key, value in outputs.items():
                bs, num_queries = value.shape[:2]

                # We flatten to compute the cost matrices in a batch
                out_bbox = value.flatten(0, 1)  # [batch_size * num_queries, 4]

                # Also concat the target labels and boxes
                if "pred_block" == key:
                    tgt_bbox = torch.cat([v["block_bbox"] for v in targets])
                else:
                    tgt_bbox = torch.cat([v["line1_bbox"] for v in targets])
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

                # Final cost matrix
                C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
                C = C.view(bs, num_queries, -1).cpu()

                if "pred_block" == key:
                    sizes = [len(v["block_bbox"]) for v in targets]
                else:
                    sizes = [len(v["line1_bbox"]) for v in targets]
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        elif "pred_line2" in outputs.keys():
            for key, value in outputs.items():
                bs, num_queries = value.shape[:2]
                # We flatten to compute the cost matrices in a batch
                out_points = value.flatten(0, 1)  # [batch_size * num_queries, 4]

                # Also concat the target labels and boxes
                tgt_points = torch.cat([v["line2_bezier"] for v in targets])
                # Compute the L1 cost between boxes
                cost_points = torch.cdist(out_points, tgt_points, p=1)

                # Final cost matrix
                C = self.cost_points * cost_points
                C = C.view(bs, num_queries, -1).cpu()

                sizes = [len(v["line2_bezier"]) for v in targets]
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]            

def build_matcher(args):
    return HungarianMatcher(cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, cost_points=args.set_cost_points)

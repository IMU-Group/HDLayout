# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def compute_iou(box1, box2):
    """
    计算两个矩形框之间的交并比(IoU)
    :param box1: 第一个框，格式为 (x1, y1, x2, y2)
    :param box2: 第二个框，格式为 (x1, y1, x2, y2)
    :return: IoU值
    """
    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])

    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)  # 添加一个小值防止除零错误
    return iou

def overlap(boxes):
    """
    计算生成框之间的overlap loss
    :param boxes: 包含n个生成框的张量, 形状为 (bs, n, 4), 每个框格式为 (cx, cy, w, h)
    :return: overlap loss
    """
    bs = boxes.size(0)
    loss_res = torch.tensor(0.0, device=boxes.device)
    for k in range(bs):
        loss = torch.tensor(0.0, device=boxes.device)
        sub_boxes = box_cxcywh_to_xyxy(boxes[k])
        num_boxes = sub_boxes.size(0)
        for i in range(num_boxes):
            for j in range(i + 1, num_boxes):
                iou = compute_iou(sub_boxes[i], sub_boxes[j])
                loss += iou
        loss /= (num_boxes * (num_boxes - 1) / 2)
        loss_res += loss
    loss_res /= bs
    return loss_res

def compute_iou_ll(box1, box2):
    """
    计算两个矩形框之间的交并比(IoU)
    :param box1: [bs, num, bbox]第一个框，格式为 (x1, y1, x2, y2)
    :param box2: [bs, num, bbox]第二个框，格式为 (x1, y1, x2, y2)
    :return: IoU值
    """
    x1_inter = torch.max(box1[:, 0], box2[:, 0])
    y1_inter = torch.max(box1[:, 1], box2[:, 1])
    x2_inter = torch.min(box1[:, 2], box2[:, 2])
    y2_inter = torch.min(box1[:, 3], box2[:, 3])

    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)  # 添加一个小值防止除零错误
    return iou

def overlap_ll(boxes):
    """
    并行计算生成框之间的overlap loss
    :param boxes: 包含n个生成框的张量, 形状为 (bs, n, 4), 每个框格式为 (cx, cy, w, h)
    :return: overlap loss
    """
    bs = boxes.size(0)
    num = boxes.size(1)
    if num == 1:
        return torch.tensor(0.0, device=boxes.device)
    num_res = num * (num - 1) // 2
    loss_res = torch.tensor(0.0, device=boxes.device)
    
    boxes = box_cxcywh_to_xyxy(boxes).permute(1, 0, 2)
    src_boxes = torch.zeros((num_res, bs, 4)).to(boxes.device)
    target_boxes = torch.zeros((num_res, bs, 4)).to(boxes.device)
    k = 0
    for i in range(boxes.size(0)):
        for j in range(i + 1, boxes.size(0)):
            src_boxes[k] = boxes[i]
            target_boxes[k] = boxes[j]
            k += 1
    iou = compute_iou_ll(src_boxes.view(-1, 4), target_boxes.view(-1, 4))
    loss_res = iou.sum() / (num_res*2)
    return loss_res
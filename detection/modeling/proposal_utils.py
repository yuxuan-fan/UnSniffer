# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List, Tuple
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION

logger = logging.getLogger(__name__)


def _is_tracing():
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        return False
    else:
        return TORCH_VERSION >= (1, 7) and torch.jit.is_tracing()


def find_top_rpn_proposals1(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.
    Args:
        proposals (list[Tensor]): 一个包含了多个张量的列表。每个张量表示一个特征图上的所有候选框（region proposals）。
        张量的形状为 (N, Hi*Wi*A, 4)，其中 N 是批量大小，Hi 和 Wi 是特征图的高度和宽度，A 是每个位置生成的候选框的数量，
        4 表示每个候选框的坐标信息（通常是左上角和右下角坐标）。

        pred_objectness_logits (list[Tensor]): 这也是一个包含多个张量的列表，
        每个张量对应一个特征图上的预测目标性（objectness）的逻辑回归分数。张量的形状为 (N, Hi*Wi*A)。

        nms_thresh (float): IoU threshold to use for NMS

        pre_nms_topk (int): 在应用NMS之前要保留的得分最高的候选框数量。当RPN在多个特征图上运行时（如在FPN中），此数字是每个特征图的数量。
        post_nms_topk (int): 在应用NMS之后要保留的得分最高的候选框数量。当RPN在多个特征图上运行时（如在FPN中），此数字是所有特征图的总数。
        min_box_size (float): 候选框的最小边长，以像素为单位。小于此长度的候选框将被删除。
        training (bool): 个布尔值，指示是否将候选框用于训练。这个参数存在是为了解决一个旧版本中的bug，如果是训练过程中使用候选框，则设为True，否则设为False。
    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. 选择得分最高的 pre_nms_topk 个候选框。
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)
        # breakpoint()
        # sort is faster than topk: https://github.com/pytorch/pytorch/issues/22812
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        ##revision##
        sample_number = len(logits_i[0])
        random_perm = torch.randperm(sample_number)
        logits_i = logits_i[:, random_perm]
        idx = random_perm.view(1, -1).repeat(len(logits_i), 1)
        ##end##
        # breakpoint()
        # logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)
        topk_idx = idx.narrow(1, 0, num_proposals_i)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


def add_ground_truth_to_proposals(gt_boxes, proposals):
    """
    这个函数的主要作用是将真实边界框添加到提议框中，以便在后续的处理中使用。
    Call `add_ground_truth_to_proposals_single_image` for all images.
    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.
    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.
    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.
    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals
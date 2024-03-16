

import logging
import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple

from torch.nn import functional as F

from torch import nn, distributions

# Detectron imports
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, cat, Conv2d, get_norm
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from detectron2.structures import Boxes, Instances, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from fvcore.nn import smooth_l1_loss

# Project imports
from inference.inference_utils import get_dir_alphas
from modeling.modeling_utils import get_probabilistic_loss_weight, clamp_log_variance, covariance_output_to_cholesky

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")


@META_ARCH_REGISTRY.register()
class ProbabilisticGeneralizedRCNN(GeneralizedRCNN):
    """
    继承GeneralizedRCNN
    Probabilistic GeneralizedRCNN class.
    实现概率通用RCNN
    这种模型可能是对传统的通用RCNN进行了改进，引入了概率性建模的思想。
    这意味着模型不仅输出单个类别的预测，还输出了与每个预测相关的概率分布。
    这样做的目的是提高模型的鲁棒性，增强其对不确定性的处理能力。

    """

    def __init__(self, cfg):
        # 调用父类的初始化方法，将配置参数 cfg 传递给父类，以完成模型的初始化。
        super().__init__(cfg)

        # Parse configs
        self.cls_var_loss = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME
        self.compute_cls_var = self.cls_var_loss != 'none'
        self.cls_var_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES

        self.bbox_cov_loss = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME
        self.compute_bbox_cov = self.bbox_cov_loss != 'none'
        self.bbox_cov_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES

        self.bbox_cov_type = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE
        # 确定了边界框协方差矩阵的维度
        if self.bbox_cov_type == 'diagonal':
            # Diagonal covariance matrix has N elements
            self.bbox_cov_dims = 4
        else:
            # Number of elements required to describe an NxN covariance matrix is
            # computed as:  (N * (N + 1)) / 2
            self.bbox_cov_dims = 10

        # 从配置参数中获取了模型的 dropout 率。
        self.dropout_rate = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        # 是否使用 dropout
        self.use_dropout = self.dropout_rate != 0.0
        # 初始化为 -1。这个变量可能用于控制 Monte Carlo Dropout 的运行次数，但在这里设置为 -1 可能表示使用默认值或者后续会被其他值覆盖。
        self.num_mc_dropout_runs = -1

        self.current_step = 0

        # Define custom probabilistic head
        # 这段代码定义了一个自定义的概率化头部，并将其设置为概率化通用RCNN模型的区域兴趣头部（ROI Heads）中的边界框预测器。
        # 这段代码用于定义并初始化概率化通用RCNN模型的概率化头部，并将模型发送到设备上以进行后续的训练或推理。
        self.roi_heads.box_predictor = ProbabilisticFastRCNNOutputLayers(
            cfg,
            self.roi_heads.box_head.output_shape,
            self.compute_cls_var,
            self.cls_var_loss,
            self.cls_var_num_samples,
            self.compute_bbox_cov,
            self.bbox_cov_loss,
            self.bbox_cov_type,
            self.bbox_cov_dims,
            self.bbox_cov_num_samples)

        # Send to device
        self.to(self.device)

    def forward(self,
                batched_inputs,
                return_anchorwise_output=False,
                num_mc_dropout_runs=-1):
        """

        总的来说，这段代码实现了概率化通用RCNN模型的前向传播逻辑，
        包括正常的训练模式下的前向传播以及 Monte Carlo Dropout 运行时的多次前向传播。
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

            return_anchorwise_output (bool): returns raw output for probabilistic inference

            num_mc_dropout_runs (int): perform efficient monte-carlo dropout runs by running only the head and
            not full neural network.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if not self.training and num_mc_dropout_runs == -1:
            if return_anchorwise_output:
                return self.produce_raw_output(batched_inputs)
            else:
                return self.inference(batched_inputs)
        elif self.training and num_mc_dropout_runs > 1:
            self.num_mc_dropout_runs = num_mc_dropout_runs
            output_list = []
            for i in range(num_mc_dropout_runs):
                output_list.append(self.produce_raw_output(batched_inputs))
            return output_list

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(
                    self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10)
            gt_instances = [x["targets"].to(self.device)
                            for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device)
                         for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances, current_step=self.current_step)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        self.current_step += 1

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def produce_raw_output(self, batched_inputs, detected_instances=None):
        """
        用于在给定输入上运行推理并返回每个提议（proposal）的原始输出，以供后续后处理使用。

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
        Returns:
            same as in :meth:`forward`.
        """

        raw_output = dict()  # 初始化一个空的字典 raw_output，用于存储原始输出。

        images = self.preprocess_image(batched_inputs)# 预处理
        features = self.backbone(images.tensor)# backbone

        if detected_instances is None:# 同RCNN
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(
                        self.device) for x in batched_inputs]
            # Create raw output dictionary
            raw_output.update({'proposals': proposals[0]})

            # 使用区域兴趣头部预测结果，
            # 并将 produce_raw_output 和 num_mc_dropout_runs 参数设置为 True 和 self.num_mc_dropout_runs，
            # 以指示需要产生原始输出和进行多次 Monte Carlo Dropout 运行。
            results, _ = self.roi_heads(
                images, features, proposals, None, produce_raw_output=True, num_mc_dropout_runs=self.num_mc_dropout_runs)
        else:
            detected_instances = [x.to(self.device)
                                  for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances)

        box_cls, box_delta, box_cls_var, box_reg_var = results

        raw_output.update({'box_cls': box_cls,
                           'box_delta': box_delta,
                           'box_cls_var': box_cls_var,
                           'box_reg_var': box_reg_var})

        return raw_output


@ROI_HEADS_REGISTRY.register()
class ProbabilisticROIHeads(StandardROIHeads):
    """
    Probabilistic ROI heads, inherit from standard ROI heads so can be used with mask RCNN in theory.
    """

    def __init__(self, cfg, input_shape):
        super(ProbabilisticROIHeads, self).__init__(cfg, input_shape)

        self.is_mc_dropout_inference = False
        self.produce_raw_output = False
        self.current_step = 0

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        num_mc_dropout_runs=-1,
        produce_raw_output=False,
        current_step=0.0,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        self.is_mc_dropout_inference = num_mc_dropout_runs > 1
        self.produce_raw_output = produce_raw_output
        self.current_step = current_step

        del images
        if self.training and not self.is_mc_dropout_inference:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training and not self.is_mc_dropout_inference:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            if self.produce_raw_output:
                return pred_instances, {}
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(
                features, pred_instances)
            return pred_instances, {}

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        负责处理对象框（bounding box）的预测逻辑。
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        # 用一个区域池化器（box_pooler）从提取的特征映射中提取与每个提议（proposal）相关的特征。
        # 提议是由 Faster R-CNN 或类似的目标检测模型生成的。
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        # 将提取的特征传递给边界框头部（box_head），用于进一步处理和提取特征
        box_features = self.box_head(box_features)
        # 使用边界框预测器（box_predictor）对处理后的特征进行预测，得到对象框的预测结果（predictions）。
        predictions = self.box_predictor(box_features)
        del box_features

        # 如果设置了 self.produce_raw_output 为 True，则直接返回原始的预测结果。
        if self.produce_raw_output:
            return predictions




        # 如果模型处于训练状态（self.training为True），则根据self.train_on_pred_boxes的值判断是否使用预测的框作为提议框。
        # 如果是，则使用box_predictor.predict_boxes_for_gt_classes方法获取预测框，并将其存储在提议中。最后返回预测框与真实框之间的损失。
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(
                            pred_boxes_per_image)
            return self.box_predictor.losses(
                predictions, proposals, self.current_step)
        else:
            # did the filtering and nms.如果模型处于推断状态（self.training 为 False），
            # 则使用 box_predictor.inference 方法进行过滤和非极大值抑制（NMS），得到最终的预测实例列表（pred_instances），并返回。
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals)
            return pred_instances


class ProbabilisticFastRCNNOutputLayers(nn.Module):
    """
    这个ProbabilisticFastRCNNOutputLayers类表示Fast R-CNN模型的输出层。它负责预测四种类型的输出：
      (1) proposal-to-detection box regression deltas ：这些是应用于建议的边界框的调整，以细化它们并更好地适应感兴趣的对象。
      (2) classification scores 这些分数指示每个提案属于每个类别的概率。
      (3) box regression deltas covariance parameters (if needed)框回归增量的协方差参数（如果需要）：如果启用（compute_bbox_cov=True），此层预测框回归增量的协方差参数。这可以捕捉到对预测的边界框调整的不确定性。
      (4) classification logits variance (if needed)如果启用（compute_cls_var=True），此层预测分类logits的方差。这可以捕捉到类别预测的不确定性。
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        compute_cls_var=False,
        compute_bbox_cov=False,
        bbox_cov_dims=4,
        cls_var_loss='none',
        cls_var_num_samples=10,
        bbox_cov_loss='none',
        bbox_cov_type='diagonal',
        dropout_rate=0.0,
        annealing_step=0,
        bbox_cov_num_samples=1000
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            compute_cls_var (bool): compute classification variance
            compute_bbox_cov (bool): compute box covariance regression parameters.
            bbox_cov_dims (int): 4 for diagonal covariance, 10 for full covariance.
            cls_var_loss (str): name of classification variance loss.
            cls_var_num_samples (int): number of samples to be used for loss computation. Usually between 10-100.
            bbox_cov_loss (str): name of box covariance loss.
            bbox_cov_type (str): 'diagonal' or 'full'. This is used to train with loss functions that accept both types.
            dropout_rate (float): 0-1, probability of drop.
            annealing_step (int): step used for KL-divergence in evidential loss to fully be functional.
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * \
            (input_shape.width or 1) * (input_shape.height or 1)

        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov

        self.bbox_cov_dims = bbox_cov_dims
        self.bbox_cov_num_samples = bbox_cov_num_samples

        self.dropout_rate = dropout_rate
        self.use_dropout = self.dropout_rate != 0.0

        self.cls_var_loss = cls_var_loss
        self.cls_var_num_samples = cls_var_num_samples

        self.annealing_step = annealing_step

        self.bbox_cov_loss = bbox_cov_loss
        self.bbox_cov_type = bbox_cov_type

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)

        # 使用 torch.nn.Linear 创建一个线性层，该层的输入大小为 input_size，输出大小为 num_classes + 1，因为它包括了目标类别数量和一个背景类别（因此加 1）。
        self.cls_score = Linear(input_size, num_classes + 1) # 分类分数层
        num_bbox_reg_classes = 1.0 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)# 边界框预测层 (bbox_pred):
        # 同样使用 torch.nn.Linear 创建一个线性层，其输入大小也为 input_size，但输出大小为 num_bbox_reg_classes * box_dim，其中 num_bbox_reg_classes 是边界框预测的类别数（可能是类别无关的边界框回归），而 box_dim 是边界框的维度。


        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if self.compute_cls_var:
            self.cls_var = Linear(input_size, num_classes + 1)
            nn.init.normal_(self.cls_var.weight, std=0.0001)
            nn.init.constant_(self.cls_var.bias, 0)

        if self.compute_bbox_cov:
            self.bbox_cov = Linear(
                input_size,
                num_bbox_reg_classes *
                bbox_cov_dims)
            nn.init.normal_(self.bbox_cov.weight, std=0.0001)
            nn.init.constant_(self.bbox_cov.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image


# 这段代码是一个类方法，用于从配置文件 (cfg) 中创建一个实例
    @classmethod
    def from_config(cls,
                    cfg,
                    input_shape,
                    compute_cls_var,
                    cls_var_loss,
                    cls_var_num_samples,
                    compute_bbox_cov,
                    bbox_cov_loss,
                    bbox_cov_type,
                    bbox_cov_dims,
                    bbox_cov_num_samples):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "compute_cls_var": compute_cls_var,
            "cls_var_loss": cls_var_loss,
            "cls_var_num_samples": cls_var_num_samples,
            "compute_bbox_cov": compute_bbox_cov,
            "bbox_cov_dims": bbox_cov_dims,
            "bbox_cov_loss": bbox_cov_loss,
            "bbox_cov_type": bbox_cov_type,
            "dropout_rate": cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE,
            "annealing_step": cfg.SOLVER.STEPS[1],
            "bbox_cov_num_samples": bbox_cov_num_samples
            # fmt: on
        }

    def forward(self, x):
        """
        这是一个神经网络模型的前向传播函数。该函数接受输入张量 x，然后执行以下操作：
        Returns:
            Tensor: Nx(K+1) logits for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
            Tensor: Nx(K+1) logits variance for each box.
            Tensor: Nx4(10) or Nx(Kx4(10)) covariance matrix parameters. 4 if diagonal, 10 if full.
        """
        if x.dim() > 2:# 如果输入张量 x 的维度大于 2，则将其展平为二维张量。
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x) # 使用神经网络模型的分类分数预测器 (cls_score) 对输入进行分类得分的预测，返回大小为 Nx(K+1) 的张量，其中 N 是样本数，K 是类别数。
        proposal_deltas = self.bbox_pred(x)# 使用神经网络模型的边界框预测器 (bbox_pred) 对输入进行边界框回归预测，返回大小为 Nx4 或 Nx(Kx4) 的张量，其中 N 是样本数，K 是类别数。

        # Compute logits variance if needed
        if self.compute_cls_var:
            score_vars = self.cls_var(x)
        else:
            score_vars = None

        # Compute box covariance if needed
        if self.compute_bbox_cov:
            proposal_covs = self.bbox_cov(x)
        else:
            proposal_covs = None

        return scores, proposal_deltas, score_vars, proposal_covs

    def losses(self, predictions, proposals, current_step=0):
        """
        predictions：forward() 方法的返回值，包括预测的类别 logits、proposal 偏移量、类别 logits 方差和 proposal 协方差。
        proposals：一个 Instances 对象的列表，表示与用于计算预测的特征匹配的提议。
        current_step：当前优化器步骤，用于具有退火组件的损失。
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
            current_step: current optimizer step. Used for losses with an annealing component.
        """
        global device

        pred_class_logits, pred_proposal_deltas, pred_class_logits_var, pred_proposal_covs = predictions

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            proposals_boxes = box_type.cat(
                [p.proposal_boxes for p in proposals])
            assert (
                not proposals_boxes.tensor.requires_grad), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            proposals_boxes = Boxes(
                torch.zeros(
                    0, 4, device=pred_proposal_deltas.device))

        no_instances = len(proposals) == 0  # no instances found

        # Compute Classification Loss
        if no_instances:
            # TODO 0.0 * pred.sum() is enough since PT1.6
            loss_cls = 0.0 * F.cross_entropy(
                pred_class_logits,
                torch.zeros(
                    0,
                    dtype=torch.long,
                    device=pred_class_logits.device),
                reduction="sum",)
        else:
            if self.compute_cls_var:#这个条件检查是否要计算分类方差。如果需要计算分类方差，
                # Compute classification variance according to:
                # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
                #这个条件检查分类方差的计算方式是否为“loss_attenuation”。这是一种根据论文
                #"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"(NIPS2017)中提出的方法。
                if self.cls_var_loss == 'loss_attenuation':
                    num_samples = self.cls_var_num_samples

                    # Compute standard deviation
                    pred_class_logits_var = torch.sqrt(
                        torch.exp(pred_class_logits_var))

                    # Produce normal samples using logits as the mean and the standard deviation computed above
                    # Scales with GPU memory. 12 GB ---> 3 Samples per anchor for
                    # COCO dataset.
                    univariate_normal_dists = distributions.normal.Normal(
                        pred_class_logits, scale=pred_class_logits_var)

                    pred_class_stochastic_logits = univariate_normal_dists.rsample(
                        (num_samples,))
                    pred_class_stochastic_logits = pred_class_stochastic_logits.view(
                        (pred_class_stochastic_logits.shape[1] * num_samples, pred_class_stochastic_logits.shape[2], -1))
                    pred_class_logits = pred_class_stochastic_logits.squeeze(
                        2)

                    # Produce copies of the target classes to match the number of
                    # stochastic samples.
                    gt_classes_target = torch.unsqueeze(gt_classes, 0)
                    gt_classes_target = torch.repeat_interleave(
                        gt_classes_target, num_samples, dim=0).view(
                        (gt_classes_target.shape[1] * num_samples, -1))
                    gt_classes_target = gt_classes_target.squeeze(1)

                    loss_cls = F.cross_entropy(
                        pred_class_logits, gt_classes_target, reduction="mean")

            elif self.cls_var_loss == 'evidential':#用了一种称为“evidential”的分类方差计算方法
                # ToDo: Currently does not provide any reasonable mAP Results
                # (15% mAP)

                # Assume dirichlet parameters are output.
                alphas = get_dir_alphas(pred_class_logits)

                # Get sum of all alphas
                dirichlet_s = alphas.sum(1).unsqueeze(1)

                # Generate one hot vectors for ground truth
                one_hot_vectors = torch.nn.functional.one_hot(
                    gt_classes, alphas.shape[1])

                # Compute loss. This loss attempts to put all evidence on the
                # correct location.
                per_instance_loss = (
                    one_hot_vectors * (torch.digamma(dirichlet_s) - torch.digamma(alphas)))

                # Compute KL divergence regularizer loss
                estimated_dirichlet = torch.distributions.dirichlet.Dirichlet(
                    (alphas - 1.0) * (1.0 - one_hot_vectors) + 1.0)
                uniform_dirichlet = torch.distributions.dirichlet.Dirichlet(
                    torch.ones_like(one_hot_vectors).type(torch.FloatTensor).to(device))
                kl_regularization_loss = torch.distributions.kl.kl_divergence(
                    estimated_dirichlet, uniform_dirichlet)

                # Compute final loss
                annealing_multiplier = torch.min(
                    torch.as_tensor(
                        current_step /
                        self.annealing_step).to(device),
                    torch.as_tensor(1.0).to(device))

                per_proposal_loss = per_instance_loss.sum(
                    1) + annealing_multiplier * kl_regularization_loss

                # Compute evidence auxiliary loss
                evidence_maximization_loss = smooth_l1_loss(
                    dirichlet_s,
                    100.0 *
                    torch.ones_like(dirichlet_s).to(device),
                    beta=self.smooth_l1_beta,
                    reduction='mean')

                evidence_maximization_loss *= annealing_multiplier

                # Compute final loss
                foreground_loss = per_proposal_loss[(gt_classes >= 0) & (
                    gt_classes < pred_class_logits.shape[1] - 1)]
                background_loss = per_proposal_loss[gt_classes ==
                                                    pred_class_logits.shape[1] - 1]

                loss_cls = (torch.mean(foreground_loss) + torch.mean(background_loss)
                            ) / 2 + 0.01 * evidence_maximization_loss
            else:
                loss_cls = F.cross_entropy(
                    pred_class_logits, gt_classes, reduction="mean")

        # Compute regression loss:
        # 在这段代码中，首先检查是否存在实例（即是否有目标检测到目标）。
        # 如果不存在实例，则将回归损失设置为零，这是因为在没有目标的情况下，没有必要计算回归损失。
        if no_instances:
            # TODO 0.0 * pred.sum() is enough since PT1.6
            # 如果存在实例，那么就会计算回归损失。具体来说，
            # 首先通过 self.box2box_transform.get_deltas() 方法计算出模型对每个提议框的预测偏移量（即回归预测）与真实标注框之间的差异。
            # 然后，根据预测的提议框偏移量的维度和是否为类别不可知来确定是否需要进行类别无关的边界框回归。
            loss_box_reg = 0.0 * smooth_l1_loss(
                pred_proposal_deltas,
                torch.zeros_like(pred_proposal_deltas),
                0.0,
                reduction="sum",
            )
        else:
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                proposals_boxes.tensor, gt_boxes.tensor
            )
            box_dim = gt_proposal_deltas.size(1)  # 4 or 5
            cls_agnostic_bbox_reg = pred_proposal_deltas.size(1) == box_dim
            device = pred_proposal_deltas.device

            bg_class_ind = pred_class_logits.shape[1] - 1

            # Box delta loss is only computed between the prediction for the gt class k
            # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
            # for non-gt classes and background.
            # Empty fg_inds produces a valid loss of zero as long as the size_average
            # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
            # and would produce a nan loss).

            # 这段代码是根据目标的类别来计算边界框回归损失（box delta loss），
            # 其中具体考虑了前景类别（foreground class）而忽略了背景类别和非目标类别。

            # 首先，通过 torch.nonzero() 函数找到所有前景类别的索引 fg_inds，这些是属于目标的类别，而不是背景或非目标类别。
            fg_inds = torch.nonzero(
                (gt_classes >= 0) & (gt_classes < bg_class_ind), as_tuple=True
            )[0]
            if cls_agnostic_bbox_reg:
                # 接着，根据模型是否为类别不可知（cls_agnostic_bbox_reg）来确定如何选择预测的提议框偏移量（pred_proposal_deltas）。
                # 如果是类别不可知的情况，则直接选择与前景类别相关的偏移量；否则，需要根据每个前景类别来选择对应的偏移量。
                gt_class_cols = torch.arange(box_dim, device=device)
            else:
                fg_gt_classes = gt_classes[fg_inds]
                # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
                # where b is the dimension of box representation (4 or 5)
                # Note that compared to Detectron1,
                # we do not perform bounding box regression for background classes.
                gt_class_cols = box_dim * \
                    fg_gt_classes[:, None] + torch.arange(box_dim, device=device)
                gt_covar_class_cols = self.bbox_cov_dims * \
                    fg_gt_classes[:, None] + torch.arange(self.bbox_cov_dims, device=device)

            # 这一行计算了一个归一化因子，即前景类别的数量。在后续的损失计算中，会将损失值除以这个因子，以便对损失进行归一化处理。
            loss_reg_normalizer = gt_classes.numel()

            # 这一行根据前景类别的索引 fg_inds 和对应的列索引 gt_class_cols 从预测的提议框偏移量中选取出需要考虑的部分。
            # 这样做是为了只计算与目标类别相关的提议框偏移量，而忽略背景类别和非目标类别的偏移量。
            pred_proposal_deltas = pred_proposal_deltas[fg_inds[:,
                                                                None], gt_class_cols]
            #  这一行根据前景类别的索引 fg_inds 从真实标注框的偏移量中选取出与前景类别相关的部分。与上一行类似，这样做是为了只考虑与目标类别相关的标注框偏移量。
            gt_proposals_delta = gt_proposal_deltas[fg_inds]

            if self.compute_bbox_cov:# 这一行首先检查是否需要计算提议框的协方差信息，如果需要，则执行下面的代码块。
                pred_proposal_covs = pred_proposal_covs[fg_inds[:,
                                                                None], gt_covar_class_cols]
                pred_proposal_covs = clamp_log_variance(pred_proposal_covs)

                if self.bbox_cov_loss == 'negative_log_likelihood':
                    if self.bbox_cov_type == 'diagonal':
                        # Ger foreground proposals.
                        _proposals_boxes = proposals_boxes.tensor[fg_inds]

                        # Compute regression negative log likelihood loss according to:
                        # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
                        loss_box_reg = 0.5 * torch.exp(-pred_proposal_covs) * smooth_l1_loss(
                            pred_proposal_deltas, gt_proposals_delta, beta=self.smooth_l1_beta)
                        loss_covariance_regularize = 0.5 * pred_proposal_covs
                        loss_box_reg += loss_covariance_regularize

                        loss_box_reg = torch.sum(
                            loss_box_reg) / loss_reg_normalizer
                    else:
                        # Multivariate Gaussian Negative Log Likelihood loss using pytorch
                        # distributions.multivariate_normal.log_prob()
                        forecaster_cholesky = covariance_output_to_cholesky(
                            pred_proposal_covs)

                        multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                            pred_proposal_deltas, scale_tril=forecaster_cholesky)

                        loss_box_reg = - \
                            multivariate_normal_dists.log_prob(gt_proposals_delta)
                        loss_box_reg = torch.sum(
                            loss_box_reg) / loss_reg_normalizer

                elif self.bbox_cov_loss == 'second_moment_matching':
                    # 如果 bbox_cov_loss 被设定为 'second_moment_matching'，
                    # 它会使用二阶矩匹配来计算回归协方差。
                    loss_box_reg = smooth_l1_loss(pred_proposal_deltas,
                                                  gt_proposals_delta,
                                                  self.smooth_l1_beta)
                    errors = (pred_proposal_deltas - gt_proposals_delta)
                    if self.bbox_cov_type == 'diagonal':
                        # Handel diagonal case
                        second_moment_matching_term = smooth_l1_loss(
                            torch.exp(pred_proposal_covs), errors ** 2, beta=self.smooth_l1_beta)
                        loss_box_reg += second_moment_matching_term
                        loss_box_reg = torch.sum(
                            loss_box_reg) / loss_reg_normalizer
                    else:
                        # Handel full covariance case
                        errors = torch.unsqueeze(errors, 2)
                        gt_error_covar = torch.matmul(
                            errors, torch.transpose(errors, 2, 1))

                        # This is the cholesky decomposition of the covariance matrix.
                        # We reconstruct it from 10 estimated parameters as a
                        # lower triangular matrix.
                        forecaster_cholesky = covariance_output_to_cholesky(
                            pred_proposal_covs)

                        predicted_covar = torch.matmul(
                            forecaster_cholesky, torch.transpose(
                                forecaster_cholesky, 2, 1))

                        second_moment_matching_term = smooth_l1_loss(
                            predicted_covar, gt_error_covar, beta=self.smooth_l1_beta, reduction='sum')
                        loss_box_reg = (
                            torch.sum(loss_box_reg) + second_moment_matching_term) / loss_reg_normalizer

                elif self.bbox_cov_loss == 'energy_loss':
                    # 这段代码主要是用于边界框回归中的能量损失模式下的损失计算，其中涉及了多元正态分布的采样和损失函数的计算。
                    forecaster_cholesky = covariance_output_to_cholesky(
                        pred_proposal_covs)

                    # Define per-anchor Distributions
                    multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                        pred_proposal_deltas, scale_tril=forecaster_cholesky)
                    # Define Monte-Carlo Samples
                    distributions_samples = multivariate_normal_dists.rsample(
                        (self.bbox_cov_num_samples + 1,))

                    distributions_samples_1 = distributions_samples[0:self.bbox_cov_num_samples, :, :]
                    distributions_samples_2 = distributions_samples[1:
                                                                    self.bbox_cov_num_samples + 1, :, :]

                    # Compute energy score
                    loss_covariance_regularize = - smooth_l1_loss(
                        distributions_samples_1,
                        distributions_samples_2,
                        beta=self.smooth_l1_beta,
                        reduction="sum") / self.bbox_cov_num_samples   # Second term

                    gt_proposals_delta_samples = torch.repeat_interleave(
                        gt_proposals_delta.unsqueeze(0), self.bbox_cov_num_samples, dim=0)

                    loss_first_moment_match = 2.0 * smooth_l1_loss(
                        distributions_samples_1,
                        gt_proposals_delta_samples,
                        beta=self.smooth_l1_beta,
                        reduction="sum") / self.bbox_cov_num_samples  # First term

                    # Final Loss
                    loss_box_reg = (
                        loss_first_moment_match + loss_covariance_regularize) / loss_reg_normalizer
                else:
                    raise ValueError(
                        'Invalid regression loss name {}.'.format(
                            self.bbox_cov_loss))

                # Perform loss annealing. Not really essential in Generalized-RCNN case, but good practice for more
                # elaborate regression variance losses.
                # 损失退火（loss annealing）操作。损失退火是一种优化技术，通常用于训练过程中的损失函数。
                # 在这里，它用于边界框回归损失（loss_box_reg）。
                standard_regression_loss = smooth_l1_loss(pred_proposal_deltas,
                                                          gt_proposals_delta,
                                                          self.smooth_l1_beta,
                                                          reduction="sum",)
                standard_regression_loss = standard_regression_loss / loss_reg_normalizer

                probabilistic_loss_weight = get_probabilistic_loss_weight(
                    current_step, self.annealing_step)

                loss_box_reg = (1.0 - probabilistic_loss_weight) * \
                    standard_regression_loss + probabilistic_loss_weight * loss_box_reg
            else:
                loss_box_reg = smooth_l1_loss(pred_proposal_deltas,
                                              gt_proposals_delta,
                                              self.smooth_l1_beta,
                                              reduction="sum",)
                loss_box_reg = loss_box_reg / loss_reg_normalizer

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.

            该函数返回两个列表：
            第一个列表包含推断结果的 Instances 对象，其中每个 Instances 对象表示一张图像的检测结果。
            第二个列表包含了一些张量，与快速 RCNN 推断相关。

        这个函数实现了将模型的预测结果转换为最终的检测结果的过程，是目标检测模型中推断阶段的关键部分。
        """

        #  首先，通过调用 predict_boxes 函数，根据模型的预测结果 predictions 和提议框 proposals 预测出边界框的坐标信息 boxes。
        boxes = self.predict_boxes(predictions, proposals)

        # 接着，通过调用 predict_probs 函数，根据模型的预测结果 predictions 和提议框 proposals 预测出每个边界框的类别得分 scores。
        scores = self.predict_probs(predictions, proposals)
        # 然后，获取每个提议框对应的图像大小，存储在 image_shapes 中。
        image_shapes = [x.image_size for x in proposals]

        # 最后，调用 fast_rcnn_inference 函数，传入预测的边界框坐标 boxes、类别得分 scores、
        # 图像大小 image_shapes，以及一些推断参数，如阈值和 NMS 阈值，执行最终的快速 RCNN 推断过程。
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        这个函数用于根据类别特定的盒子头预测提议框的边界框坐标，并将结果按照每张图片的提议框数量进行拆分返回。
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[torch.arange(
                N, dtype=torch.long, device=predict_boxes.device), gt_classes]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        这个函数用于根据模型的预测结果和提议框来预测边界框坐标。
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []

        #从模型预测结果 predictions 中获取提议框变换 proposal_deltas，忽略其他部分的预测结果。
        _, proposal_deltas, _, _ = predictions

        # 接着，计算每张图片中提议框的数量，并将提议框的信息存储在列表 proposal_boxes 中。
        # 然后将列表中的提议框张量拼接成一个张量，形状为 (N, B)，其中 N 是提议框的数量，B 是盒子的维度，通常为 4 或 5。
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor

        # 接下来，通过盒子变换 box2box_transform.apply_deltas 将预测的盒子坐标应用到提议框上，
        # 得到预测的边界框坐标 predict_boxes。该张量的形状为 (N, K * B)，其中 K 是预测的类别数。
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        #根据每张图片中提议框的数量 num_prop_per_image 对预测的边界框进行拆分，并返回拆分后的列表，列表中每个元素都是对应图片的预测边界框张量。
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """

        该函数用于根据模型的预测结果和提议框来预测每个类别的概率。
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """

        # 首先，从模型预测结果 predictions 中获取分数（scores），忽略其他部分的预测结果。
        scores, _, _, _ = predictions
        # 然后，计算每张图片中实例（目标对象）的数量，并将其存储在列表 num_inst_per_image 中。
        num_inst_per_image = [len(p) for p in proposals]
        # 接下来，根据 cls_var_loss 的值进行不同的处理。如果 cls_var_loss 为 "evidential"，
        # 则调用 get_dir_alphas 函数获取分数对应的狄利克雷分布的参数（alpha），
        # 然后计算狄利克雷分布的总和并进行归一化，得到每个类别的概率。
        if self.cls_var_loss == "evidential":
            alphas = get_dir_alphas(scores)
            dirichlet_s = alphas.sum(1).unsqueeze(1)
            # Compute probabilities
            probs = alphas / dirichlet_s
        else:
            # 如果不是 "evidential"，则使用 PyTorch 的 softmax 函数对分数进行 softmax 处理，得到每个类别的概率。
            probs = F.softmax(scores, dim=-1)
        # 最后，根据每张图片中实例的数量将概率张量 probs 拆分，并按照每张图片返回一个张量列表，
        # 其中每个张量的形状为 (Ri, K + 1)，其中 Ri 是图片 i 中预测的对象数量，K 是类别数。额外的 1 是用于表示背景类别的概率。
        return probs.split(num_inst_per_image, dim=0)


# Todo: new detectron interface required copying code. Check for better
# way to inherit from FastRCNNConvFCHead.
@ROI_BOX_HEAD_REGISTRY.register()
class DropoutFastRCNNConvFCHead(nn.Module):
    """
        A head with several 3x3 conv layers (each followed by norm & relu) and then
        several fc layers (each followed by relu) and dropout.

        这段代码定义了一个名为 DropoutFastRCNNConvFCHead 的类，用于创建一个具有多个 3x3 卷积层、多个全连接层以及 dropout 的头部结构。
        这个类注册到了 ROI_BOX_HEAD_REGISTRY 中，以便在 Detectron2 中进行使用。

        这个类的作用是构建一个具有多个卷积层、全连接层和 dropout 层的网络头部结构，用于目标检测任务中的区域兴趣框（ROI）处理。
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            conv_dims: List[int],
            fc_dims: List[int],
            conv_norm="",
            dropout_rate
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            dropout_rate (float): p for dropout layer
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self.dropout_rate = dropout_rate
        self.use_dropout = self.dropout_rate != 0.0

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (
                conv_dim,
                self._output_size[1],
                self._output_size[2])

        self.fcs = []
        self.fcs_dropout = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            fc_dropout = nn.Dropout(p=self.dropout_rate)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_dropout{}".format(k + 1), fc_dropout)
            self.fcs.append(fc)
            self.fcs_dropout.append(fc_dropout)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        #这个方法是一个类方法，用于从给定的配置 cfg 和输入形状 input_shape 创建模型的配置字典。
        # 它从配置中提取了卷积层和全连接层的数量和维度，以及规范化参数和 dropout 率，并将它们存储在一个字典中返回。
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "dropout_rate": cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        }

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer, dropout in zip(self.fcs, self.fcs_dropout):
                x = F.relu(dropout(layer(x)))
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape

        这个方法 output_shape 用于获取模型的输出特征形状。它检查 _output_size 变量的类型，
        如果是整数，表示输出特征具有固定的通道数，直接返回一个 ShapeSpec 对象，指定输出特征的通道数。如果 _output_size 是一个元组，
        表示输出特征的形状是三维的，包括通道数、高度和宽度，那么它会返回一个 ShapeSpec 对象，指定输出特征的通道数、高度和宽度。
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

# 一个用于实现简单的、通用的RCNN（Region Convolutional Neural Network）模型的文件。RCNN系列模型通常用于目标检测任务。
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from modeling.roihead_gmm import build_roi_heads
# from modeling.roihead_gmm_limit_fp_energy import build_roi_heads

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNNLogisticGMM"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNLogisticGMM(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    该组件负责从输入图像中提取特征。这些特征通常在卷积神经网络（CNN）中学习得到，并且用于后续的物体检测任务。

    2. Region proposal generation RPN
    该组件使用图像特征来生成物体区域的候选提议。这些提议通常是包含物体可能存在的矩形区域。

    3. Per-region feature extraction and prediction
    该组件负责对每个候选区域进行特征提取，并使用这些特征来预测该区域内是否包含感兴趣的物体，以及物体的类别和边界框。

    """

    # 构造函数
    @configurable
    def __init__(#构造函数
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],   # 这些是输入图像的均值
        pixel_std: Tuple[float],    # 和标准差,在模型推断时，输入图像会被标准化，即每个像素值减去均值再除以标准差，以便更好地训练模型。
        input_format: Optional[str] = None, # 描述输入图像通道含义的字符串，这在可视化过程中可能会用到。
        vis_period: int = 0,        # 可视化周期，即多少个步骤后运行一次可视化。如果设置为 0，则禁用可视化。
    ):
        """
        NOTE: this interface is experimental.
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # 这里将 pixel_mean 和 pixel_std 注册为模型的缓冲区，这意味着它们在模型的参数列表中不可训练，但在推理时会被用来标准化输入图像。
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        # 一个断言检查确保 pixel_mean 和 pixel_std 的形状一致，以避免标准化过程中的错误。
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    # 用于从配置文件中创建模型
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    # 获取模型的设备信息
    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        可视化训练过程中的图像和提议框（proposals）
        这个方法通过将原始图像与 GT 边界框和预测提议框进行叠加，从而可视化训练过程中的图像和提议框。
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """

        # 导入用于可视化的 Visualizer 类
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20   # 设置最大可视化的提议框数量。

        for input, prop in zip(batched_inputs, proposals): # 遍历批量输入和对应的提议框。
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format) # 将图像转换为 RGB 格式。
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]], iteration=0):
        """
        forward方法定义了模型的前向传播过程，用于执行推理或训练。
        实现了模型的前向传播逻辑，接收输入图像的批量数据，并返回对应的预测结果。
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
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        if not self.training:   #是否在train,不是就inference(inference定义在下面)
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        # 如果输入数据中包含 ground truth 实例信息，则将其转移到设备上，并赋值给 gt_instances。
        # 否则，将 gt_instances 设置为 None。
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        #使用bacbnoe结构提取输入图像的特征。
        features = self.backbone(images.tensor)


        # 如果模型具有提议生成器，则调用提议生成器生成提议框，并计算提议损失。
        # 否则，假设输入数据中包含预先计算的提议框，并将其转移到设备上。
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        # breakpoint()
        # 使用区域兴趣头部对特征和提议框进行处理，并计算检测器的损失。
        _, detector_losses = self.roi_heads(images, features, proposals, iteration, gt_instances)

        # 如果设置了可视化周期，则在每个周期结束时进行可视化。此处使用了 get_event_storage() 获取事件存储器，并检查当前迭代次数是否是可视化周期的倍数，如果是则调用 visualize_training 方法进行可视化。
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}

        # for key in list(detector_losses.keys()):
        #     detector_losses[key] = 0.00001 * detector_losses[key]
        # for key in list(proposal_losses.keys()):
        #     proposal_losses[key] = 0.00001 * proposal_losses[key]
        losses.update(detector_losses) # 将检测器的损失更新到总损失中。
        losses.update(proposal_losses)# 将提议生成器的损失更新到总损失中。
        # losses.update({'dummy': torch.zeros(1).cuda()})
        # losses.update({'dummy': torch.zeros(1).cuda()})

        # 最后返回losses，其中包含检测器和提议生成器的损失。
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:

            batched_inputs (list[dict]): 输入参数是一个元组，其中包含一个字典列表，每个字典对应一个图像的输入数据。

            detected_instances (None or list[Instances]): 如果不为 None，则表示已经检测到的实例信息。
            每个元素是一个 Instances 对象，包含了图像中已知的边界框和类别信息。
            如果提供了这些信息，推理过程将跳过边界框的检测，仅预测其他的每个感兴趣区域（ROI）的输出。

            do_postprocess (bool): 是否在输出结果上应用后处理。如果为 True，则返回经过后处理的结果；如果为 False，则返回原始网络输出。

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs) # 对输入图像进行预处理。
        features = self.backbone(images.tensor)         # 输入特征提取

        if detected_instances is None: # 如果 detected_instances 为 None，说明需要执行边界框检测：
            if self.proposal_generator is not None: #如果模型具有提议生成器，则使用提议生成器生成提议框。
                proposals, _ = self.proposal_generator(images, features, None)
            else:   #否则，假设输入数据中包含预先计算的提议框。
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # 使用区域兴趣头部处理特征和提议，并获取结果。
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:# 后处理
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        输入图像进行预处理,训练时调用
        标准化、填充和批处理
        Normalize, pad and batch the input images.
        """

        # 从输入字典列表中获取图像张量，并将其转移到模型所在的设备上（通常是 GPU）。
        images = [x["image"].to(self.device) for x in batched_inputs]
        # 对每张图像进行标准化处理，即将每个像素值减去均值 pixel_mean 并除以标准差 pixel_std。
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        # 使用 ImageList 类将标准化后的图像转换为一个 ImageList 对象。 类定义在当前文件夹下imagelist.py
        # ImageList 对象允许将图像批处理，并且可以根据背骨结构的要求对图像进行填充，以确保它们具有相同的大小。
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        后处理
        将实例坐标重新缩放到目标大小
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
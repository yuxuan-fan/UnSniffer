from .regnet_model import RegNet
from .regnet_model import SimpleStem, ResBottleneckBlock

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)


# train.cudnn_benchmark = True

# 这段代码定义了一个函数 build_regnet_fpn_backbone，
# 该函数用于构建一个具有 RegNet 架构和特征金字塔网络（FPN）的主干（backbone）模型，
# 并且将其注册到了 Detectron2 的 BACKBONE_REGISTRY 中。

@BACKBONE_REGISTRY.register()
def build_regnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = RegNet(
            stem_class=SimpleStem,
            stem_width=32,
            block_class=ResBottleneckBlock,
            depth=22,
            w_a=31.41,
            w_0=96,
            w_m=2.24,
            group_width=64,
            se_ratio=0.25,
            freeze_at=2,
            norm="FrozenBN",
            out_features=["s1", "s2", "s3", "s4"],
        )
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_regnetx_fpn_backbone(cfg, input_shape: ShapeSpec):
    """

    也是用于构建具有 RegNetX 架构和特征金字塔网络（FPN）的主干（backbone）模型，并将其注册到 Detectron2 的 BACKBONE_REGISTRY 中。
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = RegNet(
        stem_class=SimpleStem,
        stem_width=32,
        block_class=ResBottleneckBlock,
        depth=23,
        w_a=38.65,
        w_0=96,
        w_m=2.43,
        group_width=40,
        freeze_at=2,
        norm="FrozenBN",
        out_features=["s1", "s2", "s3", "s4"],
)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
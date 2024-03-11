from __future__ import division
from typing import Any, List, Tuple
import torch
from torch import device
from torch.nn import functional as F

from detectron2.utils.env import TORCH_VERSION


def _as_tensor(x: Tuple[int, int]) -> torch.Tensor:
    """
    An equivalent of `torch.as_tensor`, but works under tracing if input
    is a list of tensor. `torch.as_tensor` will record a constant in tracing,
    but this function will use `torch.stack` instead.

    这个函数的主要目的是提供一个与 torch.as_tensor 类似的功能，但在处理张量列表时使用 torch.stack 来确保在追踪期间的正确行为。
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x)
    if isinstance(x, (list, tuple)) and all([isinstance(t, torch.Tensor) for t in x]):
        return torch.stack(x)
    return torch.as_tensor(x)


class ImageList(object):
    """
    这段代码定义了一个名为 ImageList 的类，它是一个数据结构，用于将一组图像（可能具有不同的尺寸）存储为一个张量。
    这个类的主要目的是为了方便地管理一组图像数据，通过填充和尺寸存储功能，使它们能够以单个张量的形式进行处理。

    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.

        tensor 是一个 PyTorch 张量，代表了一组图像；
        image_sizes 是一个包含元组的列表，每个元组代表了图像的原始尺寸。
        """
        self.tensor = tensor
        self.image_sizes = image_sizes


    def __len__(self) -> int:
        # 用于返回 ImageList 实例中包含的图像数量。它返回了 self.image_sizes 的长度，即图像尺寸列表的长度
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        用于通过索引访问 ImageList 实例中的单个图像。它接受一个索引参数 idx，可以是整数或切片对象。
        根据给定的索引，它获取相应的图像尺寸，并从 self.tensor 中提取相应的图像数据，保持其原始大小
        Access the individual image in its original size.
        Args:
            idx: int or slice
        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        # 这是一个用于转换 ImageList 实例的方法，但在代码中被标记为 @torch.jit.unused，
        # 意味着它不会被 PyTorch 的 JIT（Just-In-Time）编译器使用。
        # 该方法将 ImageList 实例中的张量转换为指定的设备或数据类型，并返回一个新的 ImageList 实例，其中张量已经进行了转换。
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> device:
        # 这是一个装饰器，用于将 device 方法转换为属性，使其可以像属性一样访问而不是方法调用。该方法返回 self.tensor 的设备信息，即张量所在的设备。
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        """
        Args:
            tensors: 这是一个包含多个 PyTorch 张量的列表。每个张量代表一个图像。
            张量的形状可以是 (Hi, Wi) 或 (C_1, ..., C_K, Hi, Wi)，
            其中 Hi 是图像的高度，Wi 是图像的宽度，C_1 到 C_K 是可选的通道维度。所有张量将会被填充到具有相同形状的最大尺寸。

            size_divisibility (int): 则会在图像的高度和宽度上应用填充，以确保它们能够被 size_divisibility 整除。
            这对于一些模型来说很重要，因为它们需要输入尺寸具有特定的整除性，例如 32。如果不需要特定的整除性，可以将此参数设置为 0。

            pad_value (float): 用于填充的值。所有张量都会被填充到具有相同形状的最大尺寸，并且用 pad_value 进行填充。
        Returns:
            an `ImageList`. 函数返回一个 ImageList 实例，其中包含了从输入张量列表中构建的图像数据，以及对应的图像尺寸信息。
        """

        # 首先，代码执行了一系列断言，以确保输入的张量列表 tensors 不为空，且是一个元组或列表类型，
        # 其中每个元素都是 PyTorch 的张量对象。
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        # 接下来，代码通过循环检查每个张量的形状，确保它们的前两个维度（除去批量维度）与第一个张量的相同。
        # 这是为了确保所有张量的高度和宽度一致，因为这些张量将被堆叠成批量图像。
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape
        #     接着，代码计算了输入张量中每个图像的高度和宽度，并将它们保存在 image_sizes 中。
        # 然后，代码将图像尺寸转换为张量，并找到批量图像的最大尺寸。
        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [_as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values
        # 如果指定了 size_divisibility 参数且大于 1，代码将调整最大尺寸，以确保高度和宽度都是 size_divisibility 的倍数。
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)) // stride * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            # https://github.com/pytorch/pytorch/issues/42448
            if TORCH_VERSION >= (1, 7) and torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        # 如果只有一个张量，代码将根据最大尺寸对其进行填充，并将其扩展为批量维度。
        # 如果有多个张量，代码将创建一个全为填充值的张量，并将每个图像复制到相应的位置。
        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            batched_imgs = tensors[0].new_full(batch_shape, pad_value)
            for img, pad_img in zip(tensors, batched_imgs):
                try:
                    # breakpoint()
                    pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
                except:
                    pad_img = img
        # 最后，代码返回一个 ImageList 实例，其中包含填充后的批量图像以及每个图像的原始尺寸。
        return ImageList(batched_imgs.contiguous(), image_sizes)
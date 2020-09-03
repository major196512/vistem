import torch
from torch import nn
from torchvision.ops import roi_align as tv_roi_align
from typing import Tuple

try:
    from torchvision import __version__

    version = tuple(int(x) for x in __version__.split(".")[:2])
    USE_TORCHVISION = version >= (0, 7)  # https://github.com/pytorch/vision/pull/2438
except ImportError:  # only open source torchvision has __version__
    USE_TORCHVISION = True


if USE_TORCHVISION:
    roi_align = tv_roi_align
else:
    from torch.nn.modules.utils import _pair
    from torch.autograd import Function
    from torch.autograd.function import once_differentiable
    from vistem.ext import roi_align_forward, roi_align_backward

    class _ROIAlign(Function):
        @staticmethod
        def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
            ctx.save_for_backward(roi)
            ctx.output_size = _pair(output_size)
            ctx.spatial_scale = spatial_scale
            ctx.sampling_ratio = sampling_ratio
            ctx.input_shape = input.size()
            ctx.aligned = aligned
            output = roi_align_forward(
                input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
            )
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            (rois,) = ctx.saved_tensors
            output_size = ctx.output_size
            spatial_scale = ctx.spatial_scale
            sampling_ratio = ctx.sampling_ratio
            bs, ch, h, w = ctx.input_shape
            grad_input = roi_align_backward(
                grad_output,
                rois,
                spatial_scale,
                output_size[0],
                output_size[1],
                bs,
                ch,
                h,
                w,
                sampling_ratio,
                ctx.aligned,
            )
            return grad_input, None, None, None, None, None

    roi_align = _ROIAlign.apply


# NOTE: torchvision's RoIAlign has a different default aligned=False
class ROIAlign(nn.Module):
    def __init__(
        self, 
        output_size : Tuple[int, int], 
        spatial_scale : Tuple[float], 
        sampling_ratio : int, 
        aligned : bool = True
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(
        self, 
        input : torch.Tensor, 
        rois : torch.Tensor
    ):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr

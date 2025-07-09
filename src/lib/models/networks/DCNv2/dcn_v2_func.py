#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Function

import dcn_v2_ext as _backend
# from _ext import dcn_v2_double as _backend


class DCNv2Function(Function):

    def __init__(self):
        super(DCNv2Function, self).__init__()
        # self.stride = stride
        # self.padding = padding
        # self.dilation = dilation
        # self.deformable_groups = deformable_groups

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias, stride, padding, dilation = 1, deformable_groups = 1):
        if not input.is_cuda:
            raise NotImplementedError
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.deformable_groups = deformable_groups
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        print("weight in forward method statis", weight.shape)
        output = input.new(DCNv2Function._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new(), input.new()]
        _backend.dcn_v2_cuda_forward(input, weight,
                                     bias, ctx._bufs[0],
                                     offset, mask,
                                     output, ctx._bufs[1],
                                     weight.shape[2], weight.shape[3],
                                     ctx.stride, ctx.stride,
                                     ctx.padding, ctx.padding,
                                     ctx.dilation, ctx.dilation,
                                     ctx.deformable_groups)
        return output

    # @staticmethod
    # def setup_context(ctx, inputs, output):
    #     input, offset, mask, weight, bias, stride, padding, dilation, deformable_groups = inputs
    #     ctx.stride = stride
    #     ctx.padding = padding
    #     ctx.dilation = dilation
    #     ctx.deformable_groups = deformable_groups
    #     if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
    #         ctx.save_for_backward(input, offset, mask, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = input.new(*input.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()
        grad_mask = mask.new(*mask.size()).zero_()
        grad_weight = weight.new(*weight.size()).zero_()
        grad_bias = bias.new(*bias.size()).zero_()
        _backend.dcn_v2_cuda_backward(input, weight,
                                      bias, ctx._bufs[0],
                                      offset, mask,
                                      ctx._bufs[1],
                                      grad_input, grad_weight,
                                      grad_bias, grad_offset,
                                      grad_mask, grad_output,
                                      weight.shape[2], weight.shape[3],
                                      ctx.stride, ctx.stride,
                                      ctx.padding, ctx.padding,
                                      ctx.dilation, ctx.dilation,
                                      ctx.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias

    @staticmethod
    def _infer_shape(ctx, input, weight):
        print("input shape is ", input.shape)
        print("weight shape is ",weight.shape)
        n = input.size(0)

        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation *
                                                 (kernel_w - 1) + 1)) // ctx.stride + 1
        return (n, channels_out, height_out, width_out)


class DCNv2PoolingFunction(Function):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DCNv2PoolingFunction, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

        assert self.trans_std >= 0.0 and self.trans_std <= 1.0

    def forward(self, data, rois, offset):
        if not data.is_cuda:
            raise NotImplementedError

        output = data.new(*self._infer_shape(data, rois))
        output_count = data.new(*self._infer_shape(data, rois))
        _backend.dcn_v2_psroi_pooling_cuda_forward(data, rois, offset,
                                                   output, output_count,
                                                   self.no_trans, self.spatial_scale,
                                                   self.output_dim, self.group_size,
                                                   self.pooled_size, self.part_size,
                                                   self.sample_per_part, self.trans_std)

        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            self.save_for_backward(data, rois, offset, output_count)

        return output

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset, output_count = self.saved_tensors
        grad_input = data.new(*data.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()

        _backend.dcn_v2_psroi_pooling_cuda_backward(grad_output,
                                                    data,
                                                    rois,
                                                    offset,
                                                    output_count,
                                                    grad_input,
                                                    grad_offset,
                                                    self.no_trans,
                                                    self.spatial_scale,
                                                    self.output_dim,
                                                    self.group_size,
                                                    self.pooled_size,
                                                    self.part_size,
                                                    self.sample_per_part,
                                                    self.trans_std)
        return grad_input, None, grad_offset

    def _infer_shape(self, data, rois):
        # _, c, h, w = data.shape[:4]
        c = data.shape[1]
        n = rois.shape[0]
        return (n, self.output_dim, self.pooled_size, self.pooled_size)

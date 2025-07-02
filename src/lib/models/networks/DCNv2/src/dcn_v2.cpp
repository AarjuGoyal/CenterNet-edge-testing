#include <ATen/ATen.h>
#include <stdio.h>
#include <math.h>
#include "dcn_v2.h"

void dcn_v2_forward(at::Tensor *input, at::Tensor *weight,
                    at::Tensor *bias, at::Tensor *ones,
                    at::Tensor *offset, at::Tensor *mask,
                    at::Tensor *output, at::Tensor *columns,
                    const int pad_h, const int pad_w,
                    const int stride_h, const int stride_w,
                    const int dilation_h, const int dilation_w,
                    const int deformable_group)
{
    printf("only implemented in GPU");
}
    void dcn_v2_backward(at::Tensor *input, at::Tensor *weight,
                         at::Tensor *bias, at::Tensor *ones,
                         at::Tensor *offset, at::Tensor *mask,
                         at::Tensor *output, at::Tensor *columns,
                         at::Tensor *grad_input, at::Tensor *grad_weight,
                         at::Tensor *grad_bias, at::Tensor *grad_offset,
                         at::Tensor *grad_mask, at::Tensor *grad_output,
                         int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dilation_h, int dilation_w,
                         int deformable_group)
{
    printf("only implemented in GPU");
}
#ifndef DCN_V2_CUDA
#define DCN_V2_CUDA

#include<ATen/ATen.h>
#pragma once
int dcn_v2_cuda_forward(const at::Tensor &input,const at::Tensor &weight,
                        const at::Tensor &bias,const at::Tensor &ones,
                        const at::Tensor &offset,const at::Tensor &mask,
                        const at::Tensor &output,const at::Tensor &columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group);
int dcn_v2_cuda_backward(const at::Tensor &input,const at::Tensor &weight,
                         const at::Tensor &bias,const at::Tensor &ones,
                         const at::Tensor &offset,const at::Tensor &mask,
                         const at::Tensor &columns,
                         const at::Tensor &grad_input,const at::Tensor &grad_weight,
                         const at::Tensor &grad_bias,const at::Tensor &grad_offset,
                         const at::Tensor &grad_mask,const at::Tensor &grad_output,
                          int kernel_h, int kernel_w,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int deformable_group);

void dcn_v2_psroi_pooling_cuda_forward(at::Tensor * input, at::Tensor * bbox,
                                       at::Tensor * trans, 
                                       at::Tensor * out, at::Tensor * top_count,
                                       const int no_trans,
                                       const float spatial_scale,
                                       const int output_dim,
                                       const int group_size,
                                       const int pooled_size,
                                       const int part_size,
                                       const int sample_per_part,
                                       const float trans_std);

void dcn_v2_psroi_pooling_cuda_backward(at::Tensor * out_grad, 
                                        at::Tensor * input, at::Tensor * bbox,
                                        at::Tensor * trans, at::Tensor * top_count,
                                        at::Tensor * input_grad, at::Tensor * trans_grad,
                                        const int no_trans,
                                        const float spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const float trans_std);

// #ifdef __cplusplus
// }
// #endif

#endif
// #include <THC/THC.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
// #include<pybind11/pybind11.h>
#include <ATen/core/Tensor.h>
#include <torch/torch.h>

#include "cuda/dcn_v2_im2col_cuda.h"
#include "cuda/dcn_v2_psroi_pooling_cuda.h"
#include "dcn_v2_cuda.hpp"
typedef cudaStream_t cudaStream_t;
namespace py = pybind11;
// extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

int dcn_v2_cuda_forward(const at::Tensor &input, const at::Tensor &weight,
                        const at::Tensor &bias, const at::Tensor &ones,
                        const at::Tensor &offset, const at::Tensor &mask,
                        const at::Tensor &output, const at::Tensor &columns,
                        int kernel_h, int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const int dilation_h, const int dilation_w,
                        const int deformable_group)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 8, input, weight, bias, ones, offset, mask, output, columns));
    // only one GPU in Orin anyway

    TORCH_CHECK(input.is_contiguous(), 1, "input tensor has to be contiguous");
    TORCH_CHECK(weight.is_contiguous(), 2, "weight tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        throw("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
              kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        throw("Input shape and kernel channels wont match: (%d vs %d).",
              channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 || ones.size(0) * ones.size(1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        ones.resize_({height_out, width_out});
        ones.fill_(1);
    }
    // resize output
    output.resize_({batch, channels_out, height_out, width_out});
    // resize temporary columns
    columns.resize_({channels * kernel_h * kernel_w, 1 * height_out * width_out});

    at::Tensor input_n;
    at::Tensor offset_n;
    at::Tensor mask_n;
    at::Tensor output_n;

    for (int b = 0; b < batch; b++)
    {
        // select definition is (dim, index) hence it should be 0 dimension and iterate over batches
        input_n = input.select(0, b);   // THCudaTensor_select(state, input_n, input, 0, b);
        offset_n = offset.select(0, b); // THCudaTensor_select(state, offset_n, offset, 0, b);
        mask_n = mask.select(0, b);     // THCudaTensor_select(state, mask_n, mask, 0, b);
        output_n = output.select(0, b); //(state, output_n, output, 0, b);
        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)

        long m_ = channels_out;
        long n_ = height_out * width_out;
        long k_ = 1;
        /*Converting following cuBLAS or cuTLAS calls, as this is batched using bmm

        THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
                         THCudaTensor_data(state, ones), k_,
                         THCudaTensor_data(state, bias), k_, 0.0f,
                         THCudaTensor_data(state, output_n), n_);

         */
        auto ones_mat = ones.view({n_, k_});
        auto bias_mat = bias.view({k_, m_});
        auto bias_output = ones_mat.mm(bias_mat);                                                   // (n_ x 1) * (1 x m_) = (n_ x m_)
        output_n.copy_(bias_output.transpose(0, 1).resize_({channels_out, height_out, width_out})); // reshape({channels_out, height_out, width_out})); //ideallly the output of this should be channels_out * height_out * width_out
       
        cudaStream_t stream;
        cudaError_t cudaErr = cudaStreamCreate(&stream);
        if (cudaErr != cudaSuccess)
        {
            std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(cudaErr) << std::endl;
            return 1;
        }
        modulated_deformable_im2col_cuda(stream,
                                         input_n.data_ptr<float>(),
                                         offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group, columns.data_ptr<float>());

        cudaErr = cudaStreamDestroy(stream);
        if (cudaErr != cudaSuccess)
        {
            std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(cudaErr) << std::endl;
            return 1;
        }
        //(k * m)  x  (m * n)
        // Y = WC
        long mw_ = channels_out;
        long nw_ = height_out * width_out;
        long kw_ = channels * kernel_h * kernel_w;

        // auto columns_mat = columns.view({mw_, kw_}); //m_ * k_
        // auto weights_mat = weight.view({kw_,nw_}); // k_ * n_
        auto weights_mat = weight.view({mw_, kw_});
        auto columns_mat = columns.view({kw_, nw_});
        auto output_mat = (weights_mat.mm(columns_mat)).view({channels_out, height_out, width_out});
        output_n.add_(output_mat, 1.0);
        // output_n.addmm_(weights_mat, columns_mat, 1.0f, 1.0f);

        // output_n.resize_({channels_out, height_out, width_out}); //

        /*
        THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                         THCudaTensor_data(state, columns), n, s
                         THCudaTensor_data(state, weight), k, 1.0f,
                         THCudaTensor_data(state, output_n), n);
       writing like a normal human being
                         THCudaBlas_Sgemm(state, 'n', 'n', m, n, k, 1.0f,
                         THCudaTensor_data(state, columns), m,
                         THCudaTensor_data(state, weight), k, 1.0f,
                         THCudaTensor_data(state, output_n), m);
        */
    }
    // delete input_n;
    // delete offset_n;
    // delete mask_n;
    // delete output_n;
    return 1;
}

int dcn_v2_cuda_backward(const at::Tensor &input, const at::Tensor &weight,
                         const at::Tensor &bias, const at::Tensor &ones,
                         const at::Tensor &offset, const at::Tensor &mask,
                         const at::Tensor &columns,
                         const at::Tensor &grad_input, const at::Tensor &grad_weight,
                         const at::Tensor &grad_bias, const at::Tensor &grad_offset,
                         const at::Tensor &grad_mask, const at::Tensor &grad_output,
                         int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dilation_h, int dilation_w,
                         int deformable_group)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 13, input, weight, bias, ones, offset, mask, columns,
    //    grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output));
    TORCH_CHECK(input.is_contiguous(), 1, "input tensor has to be contiguous");
    TORCH_CHECK(weight.is_contiguous(), 2, "weight tensor has to be contiguous");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        throw("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
              kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel)
        throw("Input shape and kernel channels wont match: (%d vs %d).",
              channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 ||
        ones.size(0) * ones.size(1) < height_out * width_out)
    {
        // Resize plane and fill with ones...
        ones.resize_({height_out, width_out});
        // THCudaTensor_resize2d(state, ones, height_out, width_out);
        ones.fill_(1.0f);
    }

    grad_input.resize_({batch, channels, height, width});
    columns.resize_({channels * kernel_h * kernel_w, height_out * width_out});

    at::Tensor input_n;  //*input_n = new at::Tensor;
    at::Tensor offset_n; //*offset_n = new at::Tensor;
    at::Tensor mask_n;   //*mask_n = new at::Tensor;

    at::Tensor grad_output_n; //*grad_output_n = new at::Tensor;
    at::Tensor grad_input_n;  //*grad_input_n = new at::Tensor;
    at::Tensor grad_offset_n; //*grad_offset_n = new at::Tensor;
    at::Tensor grad_mask_n;   //*grad_mask_n = new at::Tensor;

    for (int b = 0; b < batch; b++)
    {
        input_n = input.select(0, b);             // THCudaTensor_select(state, input_n, input, 0, b);
        offset_n = offset.select(0, b);           //(state, offset_n, offset, 0, b);
        mask_n = mask.select(0, b);               // THCudaTensor_select(state, mask_n, mask, 0, b);
        grad_output_n = grad_output.select(0, b); // THCudaTensor_select(state, grad_output_n, grad_output, 0, b);
        grad_input_n = grad_input.select(0, b);   // THCudaTensor_select(state, grad_input_n, grad_input, 0, b);
        grad_offset_n = grad_offset.select(0, b); // THCudaTensor_select(state, grad_offset_n, grad_offset, 0, b);
        grad_mask_n = grad_mask.select(0, b);     // THCudaTensor_select(state, grad_mask_n, grad_mask, 0, b);

        long m_ = channels * kernel_h * kernel_w;
        long n_ = height_out * width_out;
        long k_ = channels_out;

        // A will A n_ x k_, B starts with m_ x k_, becomes k_ x m_, columns finally is n_ x m_ .

        at::Tensor grad_output_mat = grad_output_n.view({n_, k_});

        at::Tensor weight_mat = weight.view({k_, m_});

        columns.copy_(grad_output_mat.mm(weight_mat).transpose(0, 1)); //.matmul(grad_output_mat, weight_mat);
        // cublasSgemm(state, 'n', 't', n_, m_, k_, 1.0f,
        //                  grad_output_n.data_ptr<float>(), n_,
        //                  weight.data_ptr<float>(), m_, 0.0f,
        //                  columns.data_ptr<float>(), n_);
        // THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
        //                  THCudaTensor_data(state, grad_output_n), n,
        //                  THCudaTensor_data(state, weight), m, 0.0f,
        //                  THCudaTensor_data(state, columns), n);
        // above as transa is n, transb is t, alpha is 1.0f, beta is 0.0f,
        // gradient w.r.t. input coordinate data
        cudaStream_t stream1; //, stream2, stream3;
        cudaError_t cudaErr_stream1 = cudaStreamCreate(&stream1);
        if (cudaErr_stream1 != cudaSuccess)
        {
            std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(cudaErr_stream1) << std::endl;
            return 1;
        }

        modulated_deformable_col2im_coord_cuda(stream1,
                                               columns.data_ptr<float>(),
                                               input_n.data_ptr<float>(),
                                               offset_n.data_ptr<float>(),
                                               mask_n.data_ptr<float>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data_ptr<float>(),
                                               grad_mask_n.data_ptr<float>());
        // cudaStreamDestroy(stream1);
        // gradient w.r.t. input data

        // cudaError_t cudaErr_stream2 = cudaStreamCreate(&stream2);

        modulated_deformable_col2im_cuda(stream1,
                                         columns.data_ptr<float>(),
                                         offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data_ptr<float>());
        // cuStreamDestroy(stream2);
        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        // cudaError_t cudaErr = cuStreamCreate(&stream3);
        modulated_deformable_im2col_cuda(stream1,
                                         input_n.data_ptr<float>(),
                                         offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data_ptr<float>());
        // cuStreamDestroy(stream3);

        cudaErr_stream1 = cudaStreamDestroy(stream1);
        if (cudaErr_stream1 != cudaSuccess)
        {
            std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(cudaErr_stream1) << std::endl;
            return 1;
        }
        long mw_ = channels_out;
        long nw_ = channels * kernel_h * kernel_w;
        long kw_ = height_out * width_out;

        // A is k_ x n_, B is k_ x m_, grad_weight would n_ x m_
        auto columns_mat = columns.view({nw_, kw_});
        auto grad_output_n_mat = grad_output_n.view({kw_, mw_});
        auto grad_weight_inter = columns_mat.mm(grad_output_n_mat).transpose(0, 1);
        grad_weight.add_(grad_weight_inter.view({channels_out, channels, kernel_h, kernel_w}));
        // grad_weight.addmm_(columns_mat, grad_output_n_mat, 1.0f, 1.0f);

        // cublasSgemm(state, 't', 'n', n_, m_, k_, 1.0f,
        //                  columns.data(), k_,
        //                  grad_output_n.data(), k_, 1.0f,
        //                  grad_weight.data(), n_);
        // THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
        //                  THCudaTensor_data(state, columns), k_,
        //                  THCudaTensor_data(state, grad_output_n), k_, 1.0f,
        //                  THCudaTensor_data(state, grad_weight), n_);

        // gradient w.r.t. bias
        // long m_ = channels_out;
        // long k__ = height_out * width_out;
        // THCudaBlas_Sgemv(state,
        //                  't',
        //                  k_, m_, 1.0f,
        //                  THCudaTensor_data(state, grad_output_n), k_,
        //                  THCudaTensor_data(state, ones), 1, 1.0f,
        //                  THCudaTensor_data(state, grad_bias), 1);

        // Ensure grad_output_n shape is [H_out * W_out, C_out]
        auto grad_output_2d = grad_output_n.view({-1, grad_bias.size(0)}); // [H*W, C_out]
        // grad_bias += grad_output_2d.t() @ ones
        grad_bias.add_(grad_output_2d.sum(0)); // sum over H*W dimension
        // cublasSgemv(state,'t',
        //             k_, m_, 1.0f,
        //             grad_output_n.data(), k_,
        //             ones.data(), 1, 1.0f,
        //             grad_bias.data(), 1);
    }

    // delete input_n;
    // delete offset_n;
    // delete mask_n;

    // delete grad_output_n;
    // delete grad_input_n;
    // delete grad_offset_n;
    // delete grad_mask_n;
    return 1;
}

int dcn_v2_psroi_pooling_cuda_forward(at::Tensor &input, at::Tensor &bbox,
                                       at::Tensor &trans, 
                                       at::Tensor &out, at::Tensor &top_count,
                                       const int no_trans,
                                       const float spatial_scale,
                                       const int output_dim,
                                       const int group_size,
                                       const int pooled_size,
                                       const int part_size,
                                       const int sample_per_part,
                                       const float trans_std)
{
    
    TORCH_CHECK(input.is_contiguous(), 1, "input tensor has to be contiguous");
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, bbox, trans, out, top_count));

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int channels_trans = no_trans? 2 : trans.size(1);

    const int num_bbox = bbox.size(0);
    
    if (num_bbox != out.size(0))
        throw("Output shape and bbox number wont match: (%d vs %d).", 
               out.size(0), num_bbox);
    
    cudaStream_t stream1; //, stream2, stream3;
    cudaError_t cudaErr_stream1 = cudaStreamCreate(&stream1);

    if (cudaErr_stream1 != cudaSuccess)
    {
        std::cout << " Cuda strea creation failed" << std::endl;
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(cudaErr_stream1) << std::endl;
        return 1;
    }

    DeformablePSROIPoolForward(stream1,
                               input.data_ptr<float>(),
                               bbox.data_ptr<float>(),
                               trans.data_ptr<float>(),
                               out.data_ptr<float>(),
                               top_count.data_ptr<float>(),
                               batch, channels, height, width,
                               num_bbox,
                               channels_trans,
                               no_trans,
                               spatial_scale,
                               output_dim,
                               group_size,
                               pooled_size,
                               part_size,
                               sample_per_part,
                               trans_std);

    cudaErr_stream1 = cudaStreamDestroy(stream1);
    if (cudaErr_stream1 != cudaSuccess)
    {
        std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(cudaErr_stream1) << std::endl;
        return 1;
    }
    return 1;
}

void dcn_v2_psroi_pooling_cuda_backward(at::Tensor *out_grad,
                                        at::Tensor *input, at::Tensor *bbox,
                                        at::Tensor *trans, at::Tensor *top_count,
                                        at::Tensor *input_grad, at::Tensor *trans_grad,
                                        const int no_trans,
                                        const float spatial_scale,
                                        const int output_dim,
                                        const int group_size,
                                        const int pooled_size,
                                        const int part_size,
                                        const int sample_per_part,
                                        const float trans_std)
{
    TORCH_CHECK(out_grad->is_contiguous(), 0, "out_grad tensor has to be contiguous");
    TORCH_CHECK(input->is_contiguous(), 1, "input tensor has to be contiguous");
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 7, input, bbox, trans, out_grad, top_count,
    // input_grad, trans_grad));

    const int batch = input->size(0);
    const int channels = input->size(1);
    const int height = input->size(2);
    const int width = input->size(3);
    const int channels_trans = no_trans ? 2 : trans->size(1);

    const int num_bbox = bbox->size(0);
    if (num_bbox != out_grad->size(0))
        throw("Output shape and bbox number wont match: (%d vs %d).",
              out_grad->size(0), num_bbox);

    cudaStream_t stream1; //, stream2, stream3;
    cudaError_t cudaErr_stream1 = cudaStreamCreate(&stream1);
    if (cudaErr_stream1 != cudaSuccess)
    {
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(cudaErr_stream1) << std::endl;
        return;
    }
    DeformablePSROIPoolBackwardAcc(stream1,
                                   out_grad->data_ptr<float>(),
                                   input->data_ptr<float>(),
                                   bbox->data_ptr<float>(),
                                   trans->data_ptr<float>(),
                                   top_count->data_ptr<float>(),
                                   input_grad->data_ptr<float>(),
                                   trans_grad->data_ptr<float>(),
                                   batch, channels, height, width, num_bbox,
                                   channels_trans,
                                   no_trans,
                                   spatial_scale,
                                   output_dim,
                                   group_size,
                                   pooled_size,
                                   part_size,
                                   sample_per_part,
                                   trans_std);
}
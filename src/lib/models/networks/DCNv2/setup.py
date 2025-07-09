import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, _TORCH_PATH


sources = ['src/dcn_v2.cpp', 'dcn_v2_wrapper.cpp']
include_dirs=['src']
library_dirs = ['src/cuda', 'build']
defines = []
with_cuda = False

extra_objects = []
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/dcn_v2_cuda.cpp']
    include_dirs += ['src/cuda']
    defines += [('WITH_CUDA', None)]
    extra_objects += ['src/cuda/dcn_v2_im2col_cuda.cu.o']
    extra_objects += ['src/cuda/dcn_v2_psroi_pooling_cuda.cu.o']
    with_cuda = True
else:
    raise ValueError('CUDA is not available')

extra_compile_args = {'cxx': ['-std=c++17']}

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
# sources = [os.path.join(this_file, fname) for fname in sources]
include_dirs = [os.path.join(this_file, fname) for fname in include_dirs]
# include_dirs.extend([os.path.join(_TORCH_PATH, "include", "TH")])
print(include_dirs)
library_dirs = [os.path.join(this_file, fname) for fname in library_dirs]
print(library_dirs)
# extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

if with_cuda == True:
    setup(
        name='dcn_v2_ext',
        ext_modules= [
            CUDAExtension(
                name='dcn_v2_ext',
                sources=sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                define_macros=defines,
                extra_objects=extra_objects,
                extra_compile_args=extra_compile_args,
                language="c++",
                extra_link_args=['-lm','-lrt', '-lcuda']
            )
        ],
        cmdclass={
            'build_ext': BuildExtension,
        }
    )

else:
    setup(
        name="_ext.dcn_v2",
        ext_modules= [
            CppExtension(
                name="_ext.dcn_v2",
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=['-Wl,--no-as-needed', '-lm']
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
# ffi = create_extension(
#     '_ext.dcn_v2',
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects,
#     extra_compile_args=extra_compile_args
# )

# if __name__ == '__main__':
#     ffi.build()

# dcn_v2 = load(
#     name='dcn_v2',
#     sources=[pyt
#         'src/dcn_v2_cuda.cu'
#     ],
#     extra_cflags=['-O2'],
#     extra_cuda_cflags=['-O2'],
#     with_cuda=True
# )
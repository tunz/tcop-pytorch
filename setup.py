from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tcop-pytorch',
    packages=['tcop'],
    ext_modules=[
        CUDAExtension('masked_softmax_cuda', [
            'src/masked_softmax_cuda.cpp',
            'src/masked_softmax_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest"])

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name="pointcept_human_pose_serialization", ext_modules=[CUDAExtension("pointcept.models.human_pose.ops.serialization._C", ["bindings.cpp", "hilbert_cuda.cu", "hilbert_cuda_approx.cu", "morton_cuda.cu"])], cmdclass={"build_ext": BuildExtension})

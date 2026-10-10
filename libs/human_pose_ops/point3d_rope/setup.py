from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name="pointcept_human_pose_point3d_rope", ext_modules=[CUDAExtension("pointcept.models.human_pose.ops.point3d_rope._C", ["bindings.cpp", "utonia_pointrope_cuda.cu"])], cmdclass={"build_ext": BuildExtension})

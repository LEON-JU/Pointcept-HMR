from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name="pointcept_human_pose_pointnet2", ext_modules=[CUDAExtension("pointcept.models.human_pose.ops.pointnet2._C", ["pointnet2_ext.cpp", "pointnet2_ext_cuda.cu"])], cmdclass={"build_ext": BuildExtension})

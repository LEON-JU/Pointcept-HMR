from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier
from .modules import PointModule, PointModel

# Backbones used by the migration. Other legacy backbones remain importable
# from their own packages but are not eagerly loaded during human-pose startup.
from .point_transformer_v3 import *

# from .swin3d import *

from .human_pose import *

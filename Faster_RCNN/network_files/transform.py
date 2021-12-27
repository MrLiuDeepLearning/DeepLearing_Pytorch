import torch
from torch import nn, Tensor
import math
from network_files.image_list import ImageList
from torch.jit.annotations import List, Tuple, Dict, Optional
import torchvision

"""对图像进行标准化处理和Resize处理"""

class GeneralizedRCNNTransform(nn.Module):

    def __init__(self):
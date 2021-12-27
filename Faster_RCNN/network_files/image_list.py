from typing import List, Tuple
from torch import Tensor


class ImageList(object):
    def __init__(self, tensor, image_sizes):

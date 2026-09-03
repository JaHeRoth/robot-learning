import torch
from torch import Tensor
from torch.nn import Module, Transformer, TransformerEncoder, TransformerEncoderLayer, Linear, Embedding, Sequential
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import Normalize

class DiffusionPolicy:
    pass
# ml/model.py

import torch
import torch.nn as nn
from torchvision import models
from dataclasses import dataclass, field
from typing import List, OrderedDict, Optional, Tuple, Dict
from config import LOCAL_EPOCHS, FEDPROX_MU

@dataclass
class PyTorchModel:
    version: int
    model_state_dict: OrderedDict
    trained_by: List[int] = field(default_factory=list)
    logger: Optional[any] = None

    def to_device(self, model: nn.Module, device: torch.device):
        model.load_state_dict(self.model_state_dict)
        model.to(device)

    @classmethod
    def from_model(cls, model: nn.Module, version: int, trained_by: list = None):
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return cls(version=version, model_state_dict=state_dict, trained_by=trained_by or [])
    
def create_mobilenet(num_classes: int = 10, pretrained: bool = True, freeze_extractor: bool = False):
    if pretrained:
        # ImageNet 가중치 로드 (가장 좋은 가중치 자동 선택)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT 
    else:
        weights = None

    model = models.mobilenet_v3_small(weights=weights)
    
    if pretrained and freeze_extractor:
        for param in model.features.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model

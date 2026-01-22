import torch
import torch.nn as nn
from torchvision import models
from dataclasses import dataclass, field
from typing import List, OrderedDict, Optional

@dataclass
class PyTorchModel:
    version: float # [수정] 버전은 소수점(0.1) 단위로 증가하므로 float 권장
    model_state_dict: OrderedDict
    trained_by: List[int] = field(default_factory=list)
    
    def to_device(self, model: nn.Module, device: torch.device):
        """State Dict를 모델에 로드하고 디바이스로 이동"""
        model.load_state_dict(self.model_state_dict)
        model.to(device)

    @classmethod
    def from_model(cls, model: nn.Module, version: float, trained_by: list = None):
        """현재 모델의 가중치를 CPU로 복사하여 저장"""
        # [최적화] GPU 메모리 절약을 위해 무조건 CPU로 이동시켜 저장
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return cls(version=version, model_state_dict=state_dict, trained_by=trained_by or [])

def create_mobilenet(num_classes: int = 10, pretrained: bool = True, small_input: bool = True):
    """
    Args:
        small_input (bool): CIFAR-10(32x32)처럼 작은 이미지일 경우 True. 
                            첫 레이어의 Stride를 줄여 정보 손실을 막음.
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT 
    else:
        weights = None

    model = models.mobilenet_v3_small(weights=weights)

    # [수정] Pretrained 여부와 상관없이, 입력이 작으면 Stride를 줄여야 성능이 나옴
    if small_input:
        # MobileNetV3의 첫 번째 Conv 레이어 수정
        first_conv_layer = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=first_conv_layer.in_channels,
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=1, # <--- 32x32 이미지에서는 stride 1이 필수 (기존 2)
            padding=first_conv_layer.padding,
            bias=False
        )
        
    # Classifier Head 교체
    # MobileNetV3 Small의 classifier 구조: Sequential(Linear, Hardswish, Dropout, Linear)
    # 마지막 Linear(인덱스 3)를 교체합니다.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model

# [추가] ResNet용 Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # [핵심] 이것이 없으면 ResNet이 아님
        out = self.relu(out)
        return out

def create_resnet9(num_classes: int = 10):
    """
    CIFAR-10용 초경량 ResNet-9 (DavidNet 변형).
    MobileNetV3보다 학습 속도가 훨씬 빠르고 CIFAR-10에서 성능이 매우 좋음.
    """
    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool: 
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    model = nn.Sequential(
        # Prep: 32x32 -> 32x32
        conv_block(3, 64),
        
        # Layer 1: 32x32 -> 16x16
        conv_block(64, 128, pool=True),
        
        # Layer 2: Residual Block (16x16 유지)
        ResidualBlock(128), 
        
        # Layer 3: 16x16 -> 8x8
        conv_block(128, 256, pool=True),
        
        # Classifier
        nn.AdaptiveMaxPool2d((1, 1)), # [최적화] 입력 크기 상관없이 1x1로 만듦
        nn.Flatten(),
        nn.Linear(256, num_classes)
    )
    return model
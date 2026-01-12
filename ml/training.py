# ml/training.py
from typing import List, OrderedDict
from torchmetrics import JaccardIndex
import torch
import torch.nn as nn
from .model import create_mobilenet

def evaluate_model(model_state_dict, data_loader, device):
    """주어진 모델 가중치와 데이터로더로 정확도와 손실을 평가"""
    model = create_mobilenet()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    jaccard = JaccardIndex(task="multiclass", num_classes=10).to(device)

    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            jaccard.update(predicted, labels)
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    miou = jaccard.compute().item() * 100

    return accuracy, avg_loss, miou

"""
Aggregation 함수
이 부분에서 전체 모델의 성능 차이가 발생함.
연구의 핵심 부분
"""
def fed_avg(models_to_average: List[OrderedDict]) -> OrderedDict:
    """
    Federated Averaging 알고리즘을 수행.
    여러 모델의 가중치(state_dict) 리스트를 받아 각 가중치의 평균을 계산하여 반환.
    """
    if not models_to_average: return OrderedDict()
    avg_state_dict = OrderedDict()
    for key in models_to_average[0].keys():
        # 동일한 키(레이어)의 텐서들을 리스트로 모음
        tensors = [model[key].float() for model in models_to_average]
        # 텐서들을 쌓고(stack), 평균을 계산
        avg_tensor = torch.stack(tensors).mean(dim=0)
        avg_state_dict[key] = avg_tensor
    return avg_state_dict


def weighted_update(global_state_dict: OrderedDict, local_state_dict: OrderedDict, alpha: float, device: str = 'cpu') -> OrderedDict:
    """
    기존 글로벌 모델과 로컬 모델을 alpha 비율로 섞는 Weighted Update를 수행합니다.
    
    수식:
        w_new = (1 - alpha) * w_global + alpha * w_local
    
    Args:
        global_state_dict: 현재 글로벌 모델의 가중치
        local_state_dict: 업데이트할 로컬 모델의 가중치
        alpha: 로컬 모델 반영 비율 (0.0 ~ 1.0). 클수록 로컬 정보를 많이 반영.
        device: 연산을 수행할 디바이스 (예: 'cuda', 'cpu')
        
    Returns:
        OrderedDict: 업데이트된 새로운 가중치 딕셔너리 (CPU 저장)
    """
    updated_state_dict = OrderedDict()
    
    for key in global_state_dict.keys():
        # 연산을 위해 디바이스로 이동 및 float 형변환
        global_param = global_state_dict[key].to(device).float()
        
        # 로컬 모델에 해당 키가 있는지 안전장치 (보통 구조가 같으므로 생략 가능하나 안전을 위해)
        if key in local_state_dict:
            local_param = local_state_dict[key].to(device).float()
            
            # Weighted Sum 계산
            # (1 - alpha) * G + alpha * L
            updated_param = (1.0 - alpha) * global_param + alpha * local_param
            
            # 결과는 CPU로 내려서 저장 (메모리 절약)
            updated_state_dict[key] = updated_param.cpu()
        else:
            # 키가 없으면 기존 글로벌 파라미터 유지
            updated_state_dict[key] = global_state_dict[key].cpu()
            
    return updated_state_dict
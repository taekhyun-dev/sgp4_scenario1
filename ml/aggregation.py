# ml/aggregation.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, OrderedDict

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

def calculate_mixing_weight(local_ver, global_ver, local_acc, global_acc, 
                            local_data_count, avg_data_count):
    """
    레거시 코드의 동적 가중치 계산 로직 이식
    """
    BASE_ALPHA = 0.5  # 기본 반영 비율
    
    # 1. Staleness (버전 차이) 패널티
    staleness = max(0, global_ver - local_ver)
    staleness_factor = 1.0 / (1.0 + staleness * 0.5)

    performance_factor = 1.0

    # 3. Data Volume (데이터 양) 가중치
    if avg_data_count > 0:
        data_ratio = local_data_count / avg_data_count
        data_factor = np.clip(data_ratio, 0.5, 1.5)
    else:
        data_factor = 1.0

    # 최종 Alpha 계산
    final_alpha = BASE_ALPHA * staleness_factor * performance_factor * data_factor

    return min(final_alpha, 0.6), staleness_factor, performance_factor, data_factor

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
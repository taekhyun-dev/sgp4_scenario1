import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from collections import Counter
import os

def get_cifar10_loaders(num_clients: int, dirichlet_alpha: float = 0.5, 
                        data_root: str = './data', batch_size_val: int = 256, num_workers: int = 8):
    """
    CIFAR-10 데이터셋을 다운로드하고 Dirichlet 분포(Non-IID)에 따라 클라이언트별로 분할합니다.
    """
    
    # 1. CIFAR-10 전용 정규화 값 (Mean, Std)
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD  = (0.2023, 0.1994, 0.2010)

    # 2. 전처리 파이프라인 정의 (Resizing 제거 -> 32x32 원본 사용)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 데이터 증강
        transforms.RandomHorizontalFlip(),    # 데이터 증강
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    print(f"📥 [Data] CIFAR-10 데이터셋 로드 중... (Root: {data_root})")
    
    # 3. 데이터셋 다운로드 및 로드
    # (최초 실행 시 자동으로 다운로드 됩니다)
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    # 4. Dirichlet 분포를 이용한 Non-IID 데이터 분할
    print(f"⚖️ [Data] Dirichlet 분포(alpha={dirichlet_alpha})로 데이터 분할 중...")
    
    targets = np.array(train_dataset.targets) # 레이블 목록
    num_classes = 10
    
    # 각 클라이언트가 가질 데이터 인덱스 리스트
    client_indices = [[] for _ in range(num_clients)]
    
    # 클래스별로 순회하며 분배
    for k in range(num_classes):
        # 해당 클래스(k)를 가진 데이터의 인덱스들만 추출
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        
        # Dirichlet 분포로 비율 생성
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        
        # 비율을 정규화하여 개수 부족 문제 방지 (아주 적은 경우 보정)
        proportions = np.array([p * (len(idx_k) < num_clients / 10.0 and 1.0 / num_clients or 1) for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # 분할된 인덱스를 각 클라이언트에게 할당
        split_indices = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_indices[i].extend(split_indices[i])

    # 5. Subset 생성 및 데이터 통계 계산
    client_subsets = []
    total_data_count = 0
    
    for i in range(num_clients):
        # 인덱스 셔플 (클래스별로 뭉쳐있지 않게)
        np.random.shuffle(client_indices[i])
        subset = Subset(train_dataset, client_indices[i])
        client_subsets.append(subset)
        total_data_count += len(client_indices[i])

    avg_data_count = total_data_count / num_clients

    # 6. Global Validation Loader 생성
    # 검증은 배치 사이즈를 크게(256), 워커도 넉넉하게(8) 설정하여 속도 최적화
    val_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size_val, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # (디버깅) 분할 결과 요약 출력 (첫 5개 위성만)
    print(f"📊 분할 완료: 총 {total_data_count}개 학습 데이터 (위성당 평균 {avg_data_count:.1f}개)")
    for i in range(min(5, num_clients)):
        indices = client_indices[i]
        labels = [targets[idx] for idx in indices]
        counts = Counter(labels)
        print(f"  - SAT_{i}: {len(indices)} samples {dict(sorted(counts.items()))}")

    return avg_data_count, client_subsets, val_loader, train_dataset.classes
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10

def get_cifar10_loaders(num_clients, dirichlet_alpha, batch_size=128, data_root='./data', num_workers=0):
    """
    CIFAR10 데이터셋을 로드하고, 데이터 증강 및 224x224 리사이즈를 적용한 후,
    train, validation, test DataLoader를 생성하여 반환합니다.

    Args:
        batch_size (int): 데이터 로더의 배치 사이즈
        val_split (float): 전체 학습 데이터에서 검증 데이터로 사용할 비율
        data_root (str): 데이터셋을 다운로드하고 저장할 경로
        num_workers (int): 데이터 로딩에 사용할 프로세스 수. 
                         GIL 경합을 피하기 위해 0보다 큰 값으로 설정하는 것이 중요합니다.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
        transforms.RandomRotation(degrees=10),   # -10도 ~ 10도 사이로 랜덤 회전
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    temp_val_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform_test)
    test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    num_total = len(full_train_dataset)
    val_size = 5000
    train_size = num_total - val_size
    
    generator = torch.Generator().manual_seed(42)

    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, val_dataset = random_split(temp_val_dataset, [train_size, val_size], generator=generator)

    # 학습 데이터를 클라이언트별 Non-IID로 분할 ---
    num_classes = 10
    # train_dataset의 레이블 정보 가져오기
    labels = np.array(full_train_dataset.targets)[train_dataset.indices]

    client_data_indices = [[] for _ in range(num_clients)]
    indices_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    class_distribution = np.random.dirichlet([dirichlet_alpha] * num_classes, num_clients)

    # 각 클래스 데이터를 디리클레 분포에 따라 클라이언트에게 할당
    for k in range(num_classes):
        class_k_indices = indices_by_class[k]
        np.random.shuffle(class_k_indices)
        
        # 클래스 k의 데이터를 어떤 비율로 나눌지 결정
        proportions = class_distribution[:, k]
        proportions = (np.cumsum(proportions) * len(class_k_indices)).astype(int)[:-1]
        
        # 데이터를 분할하여 각 클라이언트에게 할당
        split_indices = np.split(class_k_indices, proportions)
        for i in range(num_clients):
            client_data_indices[i].extend(split_indices[i])

    # --- 5. 클라이언트별 DataLoader 생성 ---
    client_loaders = []
    client_stats_list = []
    for client_id, indices in enumerate(client_data_indices):
        client_subset = Subset(train_dataset, indices)
        loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        client_loaders.append(loader)

        # 통계 계산
        if len(indices) == 0:
            client_labels = []
            class_counts = np.zeros(num_classes, dtype=int)
        else:
            # labels (45000개) 배열에서 이 클라이언트에게 할당된 인덱스(indices)를 사용
            client_labels = labels[indices]
            class_counts = np.bincount(client_labels, minlength=num_classes)
            
        stats = {'total_samples': len(indices)}
        for c in range(num_classes):
            # CIFAR10 클래스 이름에 맞게 매핑 (선택 사항)
            # class_name = full_train_dataset.classes[c]
            # stats[class_name] = class_counts[c]
            stats[f'class_{c}'] = class_counts[c]
        client_stats_list.append(stats)

    if len(client_loaders) > 0:
        # 각 로더의 길이(배치 수)를 모두 더한 뒤 클라이언트 수로 나눔
        avg_data_count = sum(len(loader) for loader in client_loaders) / len(client_loaders)
    else:
        avg_data_count = 0

    # --- 6. 중앙 평가용 DataLoader 생성 ---
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Created {num_clients} Non-IID clients with alpha={dirichlet_alpha}.")
    print(f"Total train samples: {train_size}, Validation samples: {val_size}")

    client_stats_df = pd.DataFrame(client_stats_list)
    client_stats_df.index.name = 'Client_ID'

    # --- 1. 표(Table)로 분포 확인 ---
    print("\n--- [Client Data Distribution (Table)] ---")
    print(client_stats_df)

    return avg_data_count, client_loaders, val_loader, test_loader


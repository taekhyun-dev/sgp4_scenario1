import torch
import numpy as np
import pandas as pd
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, Dataset

# TransformedSubset 클래스는 그대로 유지
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_imagenet_loaders(num_clients, dirichlet_alpha, batch_size=128, data_root='../../../.data/imagenet/ILSVRC/Data/CLS-LOC', num_workers=4):
    """
    ImageNet (또는 ImageFolder 구조의 데이터셋) 로드 함수
    
    Args:
        data_root (str): 'train'과 'val' 폴더가 들어있는 루트 경로
    """
    abs_data_root = os.path.abspath(data_root)

    # 1. Transform 정의 (ImageNet 표준)
    # ImageNet은 이미지가 크기 때문에 224x224 리사이즈가 필수입니다.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    size = 224

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)), # 먼저 조금 크게 리사이즈
        transforms.RandomCrop(size),   # 224로 랜덤 크롭
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size),   # 중앙 크롭
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # 2. 데이터셋 로드 (ImageFolder 사용)
    train_dir = os.path.join(abs_data_root, 'train')
    val_dir = os.path.join(abs_data_root, 'val')

    if not os.path.exists(train_dir):
        print(f"❌ Error: {train_dir} not found. Please check your data path.")
        # 경로가 틀렸을 경우를 대비해 빈 값 반환
        return 0, [], None, transform_train

    # Raw Data 로드 (Transform은 나중에 적용)
    raw_train_dataset = datasets.ImageFolder(root=train_dir, transform=None)
    
    val_loader = None
    if os.path.exists(val_dir):
        test_dataset = datasets.ImageFolder(root=val_dir, transform=transform_test)
        # 검증용은 워커 조금만 써도 됨
        val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    # 3. Non-IID Dirichlet 분할 로직
    # ImageFolder는 .targets 속성에 정답(int) 리스트를 가지고 있음
    num_total = len(raw_train_dataset)
    targets = np.array(raw_train_dataset.targets) # List -> Numpy 변환 필수
    num_classes = len(raw_train_dataset.classes)  # 클래스 개수 자동 감지

    idxs_per_class = {k: np.where(targets == k)[0] for k in range(num_classes)}

    min_size = 0
    client_data_indices = [[] for _ in range(num_clients)]

    print(f"Partitioning data... (Classes: {num_classes}, Samples: {num_total})")

    # 데이터 분할 루프 (이전과 동일 로직)
    while min_size < 10:
        client_data_indices = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            idx_k = idxs_per_class[k]

            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            proportions = np.array([p * (len(idx_k) < num_clients and 1 / num_clients or p) for p in proportions])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            split_idx = np.split(idx_k, proportions)
            
            for i in range(num_clients):
                client_data_indices[i].extend(split_idx[i])
        
        min_size = min([len(idx) for idx in client_data_indices])
        if min_size < 10:
            print("  - Re-partitioning due to small client size...")

    # 4. DataLoader 생성
    client_subsets = []
    
    for indices in client_data_indices:
        client_subset_raw = Subset(raw_train_dataset, indices)
        # Transform을 적용한 Wrapper Dataset 생성
        client_dataset = TransformedSubset(client_subset_raw, transform=transform_train)
        client_subsets.append(client_dataset)

    avg_data_count = num_total / num_clients
    print(f"✅ Created {num_clients} client datasets. (Avg: {avg_data_count:.1f} samples)")

    return avg_data_count, client_subsets, val_loader, transform_train
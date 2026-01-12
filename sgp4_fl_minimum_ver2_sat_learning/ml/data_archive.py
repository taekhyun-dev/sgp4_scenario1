import torch

import torchvision

from torchvision import transforms

from torch.utils.data import DataLoader, Subset, random_split

from torchvision.datasets import CIFAR10



def get_cifar10_loaders(batch_size=128, data_root='./data', num_workers=0):
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

    # 원본 full_train_dataset에서 검증용 데이터셋을 분리하기 전에,
    # 검증용 데이터셋에 적용할 transform_test를 가진 별도의 데이터셋을 만듭니다.
    # 이렇게 하면 학습 데이터는 증강이 적용되고, 검증 데이터는 증강 없이 일관된 평가가 가능합니다.
    temp_val_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform_test)

    num_total = len(full_train_dataset)
    val_size = 5000
    train_size = num_total - val_size

    generator = torch.Generator().manual_seed(42)

    # 학습용 데이터셋은 증강이 적용된 `full_train_dataset`에서 분리합니다.
    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    # 검증용 데이터셋은 증강이 적용되지 않은 `temp_val_dataset`에서 동일한 인덱스로 분리합니다.
    _, val_dataset = random_split(temp_val_dataset, [train_size, val_size], generator=generator)

    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    # DataLoader 생성 시 num_workers를 다시 활성화합니다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"CIFAR10 DataLoaders created. (num_workers={num_workers})")

    return train_loader, val_loader, test_loader
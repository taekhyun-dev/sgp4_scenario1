# ml/training.py
from torchmetrics import JaccardIndex
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
from .model import create_mobilenet
from config import FEDPROX_MU

IMAGENET_CLASSES = 1000

def train_model(model, global_state_dict, train_loader, epochs, lr, device='cuda', sim_logger=None):
        """
        실제 PyTorch 모델 학습을 수행하는 블로킹(동기) 함수.
        asyncio 이벤트 루프를 막지 않기 위해 별도의 스레드에서 실행됩니다.
        """
        try:
            loader_length = len(train_loader)
            sim_logger.info(f"✅ [Train] 배치 개수: {loader_length}")
            if loader_length == 0:
                sim_logger.error("⚠️ DataLoader가 비어있습니다. Dataset을 확인해주세요.")
                return # 또는 다른 에러 처리
        except Exception as e:
            sim_logger.error(f"❌ DataLoader의 길이를 확인하는 중 에러 발생: {e}")

        # --- 학습 파트 ---
        model.to(device)
        model.train()
        
        for param in model.features.parameters():
            param.requires_grad = False

        # [FedProx] 비교용 글로벌 모델 (Gradient 불필요)
        # 메모리 절약을 위해 with torch.no_grad() 안에서 생성하거나 필요할 때만 로드
        global_model = create_mobilenet(num_classes=IMAGENET_CLASSES, pretrained=True)
        global_model.load_state_dict(global_state_dict)
        global_model.to(device)
        global_model.eval() # 중요: gradient가 흐르지 않도록 설정

        for param in global_model.parameters():
            param.requires_grad = False

        # # ImageNet 학습은 보통 Momentum과 Weight Decay가 필수적임
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        samples_count = 0
        scaler = GradScaler()
    
        for epoch in range(epochs):
            sim_logger.info(f"              에포크 {epoch+1}/{epochs} 진행 중...")
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                    # --- FedProx 손실 함수 수정 부분 ---
                    #     근접 항(Proximal Term) 계산: ||w - w^t||^2
                    prox_term = 0.0

                    # model.parameters() (w)와 global_model.parameters() (w^t) 비교
                    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                        # .detach()를 사용하여 w^t의 gradient가 계산되지 않도록 함
                        prox_term += torch.sum((local_param - global_param)**2)

                    # --- FedProx 손실 함수 최종 계산 부분 ---
                    #     최종 손실 계산: Loss + (mu/2) * prox_term
                    total_loss = loss + (FEDPROX_MU / 2) * prox_term

                # loss.backward()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                samples_count += labels.size(0)
            
            scheduler.step()
            
        if sim_logger:
                sim_logger.info(f"             🧠 학습 완료 (Samples: {samples_count})")

        # 메모리 정리
        model.to('cpu')
        del global_model # 명시적 삭제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return samples_count

def evaluate_model(model_state_dict, data_loader, device):
    """주어진 모델 가중치와 데이터로더로 정확도와 손실을 평가"""
    model = create_mobilenet(num_classes=IMAGENET_CLASSES, pretrained=True)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    jaccard = JaccardIndex(task="multiclass", num_classes=IMAGENET_CLASSES).to(device)

    total_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # [최적화] 추론 시에도 AMP 사용 가능 (속도 향상)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            total += labels.size(0)

            # --- Top-k Accuracy 계산 ---
            # ImageNet은 Top-5가 중요한 지표임
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            # Top-1
            correct_1 += correct[:1].reshape(-1).float().sum().item()
            # Top-5
            correct_5 += correct[:5].reshape(-1).float().sum().item()

            # mIoU (Top-1 기준)
            jaccard.update(pred[0], labels)
            
    acc1 = 100 * correct_1 / total
    acc5 = 100 * correct_5 / total
    avg_loss = total_loss / len(data_loader)
    miou = jaccard.compute().item() * 100

    model.to('cpu')
    
    return acc1, avg_loss, miou
# object/satellite.py

import asyncio
import torch
import numpy as np
from datetime import datetime, timedelta, timezone
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import setup_loggers, KST
from typing import Dict, List
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict
from skyfield.api import load, wgs84

# [수정] config에서 필요한 변수 가져오기
from config import IOT_FLYOVER_THRESHOLD_DEG, GS_FLYOVER_THRESHOLD_DEG, LOCAL_EPOCHS

# [수정] 변경된 모듈 임포트 (CIFAR-10 & ResNet-9)
from ml.data import get_cifar10_loaders
from ml.model import create_resnet9, PyTorchModel
from ml.training import train_model
from ml.aggregation import calculate_mixing_weight, weighted_update

class Satellite_Manager:
    def __init__(self, start_time: datetime, end_time: datetime, sim_logger, perf_logger):
        self.start_time = start_time
        self.end_time = end_time

        self.sim_logger = sim_logger
        self.perf_logger = perf_logger

        self.satellites: Dict[int, EarthSatellite] = {}
        self.satellite_models: Dict[int, PyTorchModel] = {}
        self.satellite_performances: Dict[int, float] = {}

        self.check_arr = defaultdict(list)

        # --- [FL 설정] ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_satellites = 50
        
        # [수정] CIFAR-10 클래스 수
        self.NUM_CLASSES = 10 

        self.sim_logger.info("CIFAR-10 데이터셋 로드 및 분할 중...")
        
        # [수정] CIFAR-10 로더 사용
        self.avg_data_count, self.client_subsets, self.val_loader, _ = get_cifar10_loaders(
            num_clients=self.num_satellites, 
            dirichlet_alpha=0.5,
            data_root='./data' # 데이터 저장 경로
        )
        self.sim_logger.info(f"데이터셋 로드 완료. 위성당 평균 데이터 수: {self.avg_data_count:.1f}")

        # 2. 글로벌 모델 초기화 (ResNet-9, Scratch Learning)
        # CIFAR-10은 작아서 Pretrained 없이 처음부터 학습해도 금방 90% 갑니다.
        self.global_model_net = create_resnet9(num_classes=self.NUM_CLASSES)
        self.global_model_net.to('cpu') 

        self.global_model_wrapper = PyTorchModel.from_model(self.global_model_net, version=0.0)
        self.best_acc = 0.0

        self.sim_logger.info("위성 관리자 생성 완료.")

    def load_constellation(self):
        tle_path = "constellation.tle"
        satellites = {}
        try:
            with open(tle_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
                i = 0
                while i < len(lines):
                    if not lines[i]: # 빈 줄 처리
                        i += 1
                        continue
                    name, line1, line2 = lines[i:i+3]
                    sat_id = int(name.replace("SAT", "").replace("_", ""))
                    satellites[sat_id] = EarthSatellite(line1, line2, name)
                    i += 3
            self.satellites = satellites
        except Exception as e:
            self.sim_logger.error(f"TLE 파일 로드 실패: {e}")
            raise e

    async def run(self):
        self.sim_logger.info("위성 관리자 운영 시작.")
        self.load_constellation()

        # 초기 모델 배포
        for sat_id in self.satellites.keys():
            self.satellite_models[sat_id] = PyTorchModel.from_model(self.global_model_net, version=0.0)
            self.satellite_performances[sat_id] = 0.0

        await self.propagate_orbit(self.start_time, self.end_time)
        self.sim_logger.info(f"궤도 전파 완료 ({len(self.times)} steps).")

        await self.check_iot_comm()
        await self.check_gs_comm()
        self.sim_logger.info("모든 통신 스케줄 계산 완료.")

        await self.manage_fl_process()
        self.sim_logger.info("모든 시뮬레이션 종료.")

    async def propagate_orbit(self, start_time, end_time):
        # [최적화] step을 10초로 유지 (너무 크면 통신 놓침, 너무 작으면 느림)
        step = timedelta(seconds=10)
        self.times = []
        curr = start_time
        while curr < end_time:
            self.times.append(curr)
            curr += step

        ts = load.timescale() 
        self.t_vector = ts.from_datetimes(self.times)

    async def check_iot_comm(self):
        self.sim_logger.info("IoT 통신 가능 시간 분석 시작...")

        iot_devices = [
            {"name": "Amazon_Forest", "loc": wgs84.latlon(-3.47, -62.37, elevation_m=100)},
            {"name": "Great_Barrier_Reef", "loc": wgs84.latlon(-18.29, 147.77, elevation_m=0)},
            {"name": "Abisko Tundra", "loc": wgs84.latlon(68.35, 18.79, elevation_m=420)},
        ]     

        for iot in iot_devices:
            for sat_id, satellite in self.satellites.items():
                difference = satellite - iot['loc']
                topocentric = difference.at(self.t_vector)
                alt, _, _ = topocentric.altaz()

                visible_indices = np.where(alt.degrees > IOT_FLYOVER_THRESHOLD_DEG)[0]
                if len(visible_indices) == 0: continue

                breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
                windows = np.split(visible_indices, breaks)

                for window in windows:
                    start_idx = window[0]
                    end_idx = window[-1]
                    
                    start_time = self.times[start_idx]
                    end_time = self.times[end_idx]
                    duration = (end_time - start_time).total_seconds()
                    
                    if duration == 0: duration = 10

                    event = {
                        "type": "IOT_TRAIN",
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "target": iot['name']
                    }
                    self.check_arr[sat_id].append(event)

    async def check_gs_comm(self):
        self.sim_logger.info("지상국 통신 가능 시간 분석 시작...")
        gs = {"name": "Ground Station", "loc": wgs84.latlon(37.5665, 126.9780, elevation_m=34)}

        for sat_id, satellite in self.satellites.items():
            difference = satellite - gs['loc']
            topocentric = difference.at(self.t_vector)
            alt, _, _ = topocentric.altaz()

            visible_indices = np.where(alt.degrees > GS_FLYOVER_THRESHOLD_DEG)[0]
            if len(visible_indices) == 0: continue

            breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
            windows = np.split(visible_indices, breaks)

            for window in windows:
                start_idx = window[0]
                end_idx = window[-1]
                
                start_time = self.times[start_idx]
                end_time = self.times[end_idx]
                duration = (end_time - start_time).total_seconds()
                
                if duration == 0: duration = 10

                event = {
                    "type": "GS_AGGREGATE",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "target": gs['name']
                }
                self.check_arr[sat_id].append(event)

    def _evaluate_direct(self, model, data_loader, sat_id, version, stage):
        model.to(self.device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0

        # CSV 로깅
        self.perf_logger.info(
            f"{datetime.now(KST).isoformat()},{stage},{sat_id},{version:.2f},N/A,{acc:.4f},{avg_loss:.6f},0.0000"
        )
        return acc, avg_loss

    async def manage_fl_process(self):
        self.sim_logger.info("\n=== 연합 학습 시뮬레이션 시작 (Time-Ordered) ===")
        
        # 1. 모든 위성의 이벤트를 하나로 모으기 (Time-Ordered Execution)
        all_events = []
        for sat_id, events in self.check_arr.items():
            for event in events:
                event['sat_id'] = sat_id
                all_events.append(event)
        
        # 2. 시작 시간 기준으로 전체 정렬
        all_events.sort(key=lambda x: x['start_time'])
        [self.sim_logger.info(i) for i in all_events]
        
        self.sim_logger.info(f"📅 총 {len(all_events)}개의 이벤트가 시간순으로 정렬되었습니다.")
        
        # [수정] ResNet-9 임시 모델 생성 (빈 껍데기)
        temp_model = create_resnet9(num_classes=self.NUM_CLASSES)

        for i, event in enumerate(all_events):
            sat_id = event['sat_id']
            current_local_wrapper = self.satellite_models[sat_id]
            
            if event['type'] == 'IOT_TRAIN':
                self.sim_logger.info(f"\n📡 [Time: {event['start_time'].strftime('%m-%d %H:%M')}] SAT_{sat_id} : IoT 학습 시작")
                
                epochs = LOCAL_EPOCHS
                loader_idx = sat_id % len(self.client_subsets)
                dataset = self.client_subsets[loader_idx]

                # [최적화] 고사양 PC에 맞춘 DataLoader 설정
                train_loader = DataLoader(
                        dataset, 
                        batch_size=128,  # 32코어 PC면 128 추천
                        shuffle=True, 
                        num_workers=8,   # 8~16 추천
                        pin_memory=True, 
                        persistent_workers=False 
                )

                current_local_wrapper.to_device(temp_model, device='cpu')

                # [중요] 처음부터 학습(Scratch)이므로 LR을 0.001로 설정 (ImageNet Fine-tuning은 1e-5였음)
                train_model(
                    model=temp_model, 
                    global_state_dict=self.global_model_wrapper.model_state_dict,
                    train_loader=train_loader, 
                    epochs=epochs, 
                    lr=0.005,  # <--- ResNet9 Scratch 학습용 LR (Adam 기본값)
                    device=self.device,
                    sim_logger=self.sim_logger
                )

                # 버전 증가
                next_version = round(current_local_wrapper.version + 0.1, 1)

                acc, loss = self._evaluate_direct(
                    temp_model, 
                    self.val_loader, 
                    sat_id=sat_id, 
                    version=next_version, 
                    stage="LOCAL_TRAIN"
                )
                self.satellite_performances[sat_id] = acc 

                current_local_wrapper = PyTorchModel.from_model(
                    temp_model, 
                    version=next_version
                )
                self.satellite_models[sat_id] = current_local_wrapper

                self.sim_logger.info(f"   ✅ [Result] Acc: {acc:.2f}%, v{current_local_wrapper.version:.1f}")
                
            elif event['type'] == 'GS_AGGREGATE':
                self.sim_logger.info(f"\n📡 [Time: {event['start_time'].strftime('%m-%d %H:%M')}] SAT_{sat_id} : 지상국 접속")

                # [정책] 글로벌 모델 버전 차이가 1.0 이상이면 그냥 다운로드 (동기화)
                # ResNet9은 학습이 빠르므로 너무 오래된 모델은 병합하지 않고 덮어씁니다.
                if self.global_model_wrapper.version > current_local_wrapper.version + 5.0:
                    current_local_wrapper = PyTorchModel.from_model(
                        self.global_model_net, 
                        version=self.global_model_wrapper.version
                    )
                    self.satellite_models[sat_id] = current_local_wrapper
                    self.sim_logger.info(f"   📥 Global Model Downloaded (v{self.global_model_wrapper.version})")
                    continue 

                # Aggregation 진행
                local_acc = self.satellite_performances[sat_id]
                loader_idx = sat_id % len(self.client_subsets)
                local_data_count = len(self.client_subsets[loader_idx])

                alpha, _, _, _ = calculate_mixing_weight(
                    local_ver=current_local_wrapper.version,
                    global_ver=self.global_model_wrapper.version,
                    local_acc=local_acc,
                    global_acc=self.best_acc,
                    local_data_count=local_data_count,
                    avg_data_count=self.avg_data_count
                )

                alpha = 0.2

                new_state_dict = weighted_update(
                    self.global_model_wrapper.model_state_dict,
                    current_local_wrapper.model_state_dict,
                    alpha
                )

                # 글로벌 버전 업데이트 (정수 단위)
                new_version = int(self.global_model_wrapper.version) + 1.0

                temp_model.load_state_dict(new_state_dict)

                g_acc, g_loss = self._evaluate_direct(
                    temp_model, 
                    self.val_loader,
                    sat_id="GS",     
                    version=new_version,
                    stage="GLOBAL_TEST"
                )

                if g_acc > self.best_acc:
                    previous_best = self.best_acc
                    self.best_acc = g_acc
                    
                    save_dir = Path("./checkpoints")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"global_v{int(new_version)}_acc{g_acc:.2f}.pth"
                    save_path = save_dir / filename
                    
                    checkpoint = {
                        'model_state_dict': new_state_dict,
                        'version': new_version,
                        'accuracy': g_acc,
                        'loss': g_loss,
                        'timestamp': datetime.now().isoformat(),
                        'description': f"Best Global Model (ResNet9) at Round {new_version}"
                    }
                    torch.save(checkpoint, save_path)
                    self.sim_logger.info(f"   💾 [Save] New Best Model! ({previous_best:.2f}% -> {self.best_acc:.2f}%)")

                # Global Wrapper 갱신
                self.global_model_wrapper = PyTorchModel(
                    version=new_version,
                    model_state_dict=new_state_dict, 
                    trained_by=self.global_model_wrapper.trained_by + [sat_id]
                )
                self.global_model_net.load_state_dict(new_state_dict)

                self.sim_logger.info(f"   ⚡ [Aggregation] SAT_{sat_id} (Alpha: {alpha:.4f}) -> Global v{new_version:.1f} (Acc: {g_acc:.2f}%)")
                
                # 위성 동기화
                current_local_wrapper = PyTorchModel.from_model(temp_model, version=new_version)
                self.satellite_models[sat_id] = current_local_wrapper

        self.sim_logger.info("\n=== 시뮬레이션 종료 ===")
        self.sim_logger.info(f"Final Global Model Accuracy: {self.best_acc:.2f}%")

def main():
    try:
        start_time = datetime.now(timezone.utc)
        sim_logger, perf_logger = setup_loggers()
        # 14일 시뮬레이션
        sat_manager = Satellite_Manager(start_time, start_time + timedelta(days=30), sim_logger, perf_logger)
        asyncio.run(sat_manager.run())
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
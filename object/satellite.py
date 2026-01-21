import asyncio
import torch
import numpy as np
from datetime import datetime, timedelta
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import setup_loggers, KST
from typing import Dict
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict
from config import IOT_FLYOVER_THRESHOLD_DEG, GS_FLYOVER_THRESHOLD_DEG, LOCAL_EPOCHS
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84
from ml.data import get_imagenet_loaders
from ml.model import create_mobilenet, PyTorchModel
from ml.training import train_model
from ml.aggregation import calculate_mixing_weight, weighted_update

class Satellite_Manager:
    def __init__(self, start_time: datetime, end_time: datetime, sim_logger, perf_logger):
        self.start_time = start_time
        self.end_time = end_time

        self.sim_logger = sim_logger
        self.perf_logger = perf_logger

        self.satellites:Dict[int, EarthSatellite] = {}
        self.satellite_models: Dict[int, PyTorchModel] = {}
        self.satellite_performances: Dict[int, float] = {}

        self.check_arr = defaultdict(list)

        # --- [FL 설정] ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_satellites = 50

        self.NUM_CLASSES = 1000

        self.sim_logger.info("데이터셋 로드 및 분할 중...")
        self.avg_data_count, self.client_subsets, self.val_loader, _ = get_imagenet_loaders(
            num_clients=self.num_satellites, dirichlet_alpha=0.5
        )
        self.sim_logger.info(f"데이터셋 로드 및 분할 완료. 위성당 평균 데이터 수: {self.avg_data_count}")
        # 2. 글로벌 모델 초기화 (ImageNet Pretrained)
        self.global_model_net = create_mobilenet(num_classes=self.NUM_CLASSES, pretrained=True)
        self.global_model_net.to('cpu') # 평소엔 CPU에 대기

        self.global_model_wrapper = PyTorchModel.from_model(self.global_model_net, version=0)
        self.best_acc = 0.0

        self.sim_logger.info("위성 관리자 생성 완료.")

    def load_constellation(self) -> Dict[int, EarthSatellite]:
            tle_path = "constellation.tle"
            satellites = {}
            with open(tle_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]; i = 0
                while i < len(lines):
                    name, line1, line2 = lines[i:i+3]
                    sat_id = int(name.replace("SAT", "").replace("_", "")) # 이름 파싱 예외처리 강화
                    satellites[sat_id] = EarthSatellite(line1, line2, name)
                    i += 3
            self.satellites = satellites

    async def run(self):
        self.sim_logger.info("위성 관리자 운영 시작.")
        # step = timedelta(days=1)
        self.load_constellation()

        for sat_id in self.satellites.keys():
            self.satellite_models[sat_id] = PyTorchModel.from_model(self.global_model_net, version=0)
            self.satellite_performances[sat_id] = 0.0

        await self.propagate_orbit(self.start_time, self.end_time)
        self.sim_logger.info(f"궤도 전파 완료 ({len(self.times)} steps).")

        await self.check_iot_comm()
        await self.check_gs_comm()
        self.sim_logger.info("모든 통신 스케줄 계산 완료.")

        await self.manage_fl_process()
        self.sim_logger.info("모든 시뮬레이션 종료.")

    async def propagate_orbit(self, start_time, end_time):
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

                if len(visible_indices) == 0:
                    continue

                breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
                windows = np.split(visible_indices, breaks)

                self.sim_logger.info(f"📡 [SAT_{sat_id} <-> {iot['name']}] 총 {len(windows)}회 접속 발생")

                for i, window in enumerate(windows):
                    start_idx = window[0]
                    end_idx = window[-1]
                    
                    start_time = self.times[start_idx]
                    end_time = self.times[end_idx]
                    duration = (end_time - start_time).total_seconds()
                    
                    # 10초 미만(점 하나)인 경우 duration이 0으로 나올 수 있으므로 보정 (선택사항)
                    if duration == 0:
                        duration = 10

                    event = {
                        "type": "IOT_TRAIN",
                        "start_time": self.times[start_idx],
                        "end_time": self.times[end_idx],
                        "duration": duration,
                        "target": iot['name']
                    }
                    self.check_arr[sat_id].append(event)
                    self.sim_logger.info(f"  └─ #{i+1}: {start_time.strftime('%H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')} ({duration} 유지)")

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

            self.sim_logger.info(f"📡 [SAT_{sat_id} <-> {gs['name']}] 총 {len(windows)}회 접속 발생")

            for i, window in enumerate(windows):
                start_idx = window[0]
                end_idx = window[-1]
                
                start_time = self.times[start_idx]
                end_time = self.times[end_idx]
                duration = (end_time - start_time).total_seconds()
                
                # 10초 미만(점 하나)인 경우 duration이 0으로 나올 수 있으므로 보정 (선택사항)
                if duration == 0:
                    duration = 10

                event = {
                    "type": "GS_AGGREGATE",
                    "start_time": self.times[start_idx],
                    "end_time": self.times[end_idx],
                    "duration": duration,
                    "target": gs['name']
                }
                self.check_arr[sat_id].append(event)
                self.sim_logger.info(f"  └─ #{i+1}: {start_time.strftime('%H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')} ({duration} 유지)")

    def _evaluate_direct(self, model, data_loader, sat_id, version, stage):
        """
        [최적화] 모델 재생성 없이 기존 모델 객체로 바로 평가
        (evaluate_model 함수를 호출하면 매번 create_mobilenet을 수행하므로 비효율적)
        """
        model.to(self.device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        # ImageNet은 클래스가 많아 mIoU 계산이 비쌀 수 있음
        # 필요하다면 mIoU 계산 부분은 제거하거나 빈도로 조절
        
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

        self.perf_logger.info(
            f"{datetime.now(KST).isoformat()},{stage},{sat_id},{version},N/A,{acc:.4f},{avg_loss:.6f},0.0000"
        )
        return acc, avg_loss

    async def manage_fl_process(self):
        self.sim_logger.info("\n=== 연합 학습 시뮬레이션 시작 ===")
        
        temp_model = create_mobilenet(num_classes=self.NUM_CLASSES, pretrained=True)

        for sat_id in self.satellites.keys():
            events = self.check_arr[sat_id]
            if not events: continue

            events = sorted(events, key=lambda x: x['start_time'])
            current_local_wrapper = self.satellite_models[sat_id]

            self.sim_logger.info(f"\n📡 [SAT_{sat_id}] 스케줄 처리 (이벤트 {len(events)}건)")

            for event in events:
                if event['type'] == 'IOT_TRAIN':
                    # Option 1. 멀티 에포크 진행 시나리오
                    # duration = event['duration']
                    # epochs = max(1, int(duration // 10))

                    # Option 2. 싱글 에포크 진행 시나리오
                    epochs = LOCAL_EPOCHS

                    loader_idx = sat_id % len(self.client_subsets)
                    dataset = self.client_subsets[loader_idx]

                    train_loader = DataLoader(
                            dataset, batch_size=64, shuffle=True, 
                            num_workers=0, pin_memory=True, persistent_workers=False # 일회용이므로 False 추천
                    )

                    current_local_wrapper.to_device(temp_model, device='cpu')

                    train_model(
                        model=temp_model, 
                        global_state_dict=self.global_model_wrapper.model_state_dict, # FedProx 비교용
                        train_loader=train_loader, 
                        epochs=epochs, 
                        lr=0.0003, 
                        device=self.device,
                        sim_logger=self.sim_logger # 로거 전달
                    )

                    acc, loss = self._evaluate_direct(
                        temp_model, 
                        self.val_loader, 
                        sat_id=sat_id, 
                        version=current_local_wrapper.version + 0.1, # 예측되는 새 버전
                        stage="LOCAL_TRAIN"
                    )
                    self.satellite_performances[sat_id] = acc # 성능 기록

                    # 학습된 상태 저장 (GPU -> CPU)
                    current_local_wrapper = PyTorchModel.from_model(
                        temp_model, 
                        version=current_local_wrapper.version + 0.1 # 버전 소폭 증가
                    )
                    self.satellite_models[sat_id] = current_local_wrapper

                    self.sim_logger.info(f"🛰️ [SAT_{sat_id}] Local Train (Acc: {acc:.2f}%, v{current_local_wrapper.version:.1f})")
                    
                elif event['type'] == 'GS_AGGREGATE':
                    # 글로벌 모델이 로컬보다 최신이면 -> 로컬이 글로벌을 다운로드 (덮어쓰기)
                    if self.global_model_wrapper.version > current_local_wrapper.version + 1.0:
                        current_local_wrapper = PyTorchModel.from_model(
                            self.global_model_net, 
                            version=self.global_model_wrapper.version
                        )
                        self.satellite_models[sat_id] = current_local_wrapper
                        self.sim_logger.info(f"📥 [SAT_{sat_id}] Global Model Downloaded (v{self.global_model_wrapper.version})")
                        continue # 이번 턴은 Aggregation 없이 다운로드만 하고 종료

                    # Aggregation 진행
                    local_acc = self.satellite_performances[sat_id]
                    loader_idx = sat_id % len(self.client_subsets)
                    local_data_count = len(self.client_subsets[loader_idx])

                    # [레거시 전략] Mixing Weight 계산
                    alpha, _, _, _ = calculate_mixing_weight(
                        local_ver=current_local_wrapper.version,
                        global_ver=self.global_model_wrapper.version,
                        local_acc=local_acc,
                        global_acc=self.best_acc,
                        local_data_count=local_data_count,
                        avg_data_count=self.avg_data_count
                    )

                    # 가중치 업데이트 (Weighted Average)
                    new_state_dict = weighted_update(
                        self.global_model_wrapper.model_state_dict,
                        current_local_wrapper.model_state_dict,
                        alpha
                    )

                    # 글로벌 모델 업데이트 적용
                    new_version = int(self.global_model_wrapper.version) + 1

                    # State Dict를 실제 모델 객체에 로드해서 평가 진행
                    temp_model.load_state_dict(new_state_dict)

                    # 글로벌 모델 평가
                    g_acc, g_loss = self._evaluate_direct(
                        temp_model, 
                        self.val_loader,
                        sat_id="GS",     # 지상국(Global) 표시
                        version=new_version,
                        stage="GLOBAL_TEST"
                    )

                    # 최고 성능 갱신 여부 확인
                    if g_acc > self.best_acc:
                        previous_best = self.best_acc
                        self.best_acc = g_acc
                        # (선택) 모델 파일 저장 로직 추가 가능

                        # [추가됨] 1. 저장 디렉토리 생성
                        save_dir = Path("./checkpoints")
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # [추가됨] 2. 파일명 설정 (예: global_v1_acc75.23.pth)
                        filename = f"global_v{new_version}_acc{g_acc:.2f}.pth"
                        save_path = save_dir / filename
                        
                        # [추가됨] 3. 체크포인트 딕셔너리 구성
                        # 나중에 resume하거나 분석할 때 필요한 정보들을 함께 저장합니다.
                        checkpoint = {
                            'model_state_dict': new_state_dict,  # 모델 가중치
                            'version': new_version,              # 모델 버전
                            'accuracy': g_acc,                   # 달성 정확도
                            'loss': g_loss,                      # 달성 Loss
                            'timestamp': datetime.now().isoformat(), # 저장 시간
                            'description': f"Best Global Model at Round {new_version}"
                        }
                        
                        # [추가됨] 4. 파일 저장
                        torch.save(checkpoint, save_path)
                        
                        self.sim_logger.info(f"💾 [Save] New Best Model Saved! ({previous_best:.2f}% -> {self.best_acc:.2f}%) Path: {save_path}")

                    # Global Wrapper 갱신
                    self.global_model_wrapper = PyTorchModel(
                        version=new_version,
                        model_state_dict=new_state_dict, # 이미 CPU에 있음
                        trained_by=self.global_model_wrapper.trained_by + [sat_id]
                    )
                    # 메모리상의 메인 모델 객체도 동기화 (다음 다운로드를 위해)
                    self.global_model_net.load_state_dict(new_state_dict)

                    self.sim_logger.info(f"⚡ [GS Aggregation] SAT_{sat_id} (Alpha: {alpha:.4f}) -> Global v{new_version} (Acc: {g_acc:.2f}%)")
                    
                    # Aggregation 후, 위성도 최신 글로벌 모델로 업데이트 (동기화)
                    current_local_wrapper = PyTorchModel.from_model(temp_model, version=new_version)
                    self.satellite_models[sat_id] = current_local_wrapper

        self.sim_logger.info("\n=== 시뮬레이션 종료 ===")
        self.sim_logger.info(f"Final Global Model Accuracy: {self.best_acc:.2f}%")

def main():
    try:
        start_time = datetime.now(timezone.utc)
        sim_logger, perf_logger = setup_loggers()
        sat_manager = Satellite_Manager(start_time, start_time + timedelta(days=1),sim_logger, perf_logger)
        asyncio.run(sat_manager.run())
    except KeyboardInterrupt:
        sim_logger.info("\n시뮬레이션을 종료합니다.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        # 예기치 않은 에러 발생 시 로깅
        sim_logger.error(f"\n시뮬레이션 중 치명적인 에러 발생: {e}", exc_info=True)
        
if __name__ == "__main__":
    main()
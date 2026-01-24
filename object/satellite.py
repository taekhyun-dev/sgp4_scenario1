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
        self.sim_logger.info("\n=== 연합 학습 시뮬레이션 시작 (Time-Ordered) SYNC WAY ===")

        MIN_PARTICIPANTS = 10

        agg_buffer: Dict[int, dict] = {}
        sat_status = defaultdict(lambda: 'IDLE')
        
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
        
        # 초기 모델 (ResNet-9)
        temp_model = create_resnet9(num_classes=self.NUM_CLASSES)

        for i, event in enumerate(all_events):
            sat_id = event['sat_id']
            current_local_wrapper = self.satellite_models[sat_id]
            global_version = self.global_model_wrapper.version
                
            # -----------------------------------------------------------
            # [이벤트 1] 지상국 접속 (모델 다운로드 OR 결과 업로드)
            # -----------------------------------------------------------
            if event['type'] == 'GS_AGGREGATE':
                self.sim_logger.info(f"\n📡 [Time: {event['start_time'].strftime('%m-%d %H:%M')}] SAT_{sat_id} : 지상국 접속")

                # Case A: 학습된 모델이 있어서 제출(Upload) 하러 옴
                # 조건: 상태가 TRAINED이고, 가지고 있는 모델이 현재 글로벌 모델(의 파생)일 때
                if sat_status[sat_id] == 'TRAINED' and current_local_wrapper.version == global_version:
                    self.sim_logger.info(f"   ⬆️ [SAT_{sat_id}] Uploading model to buffer...")

                    # 버퍼에 추가 (이미 제출했으면 덮어쓰기)
                    agg_buffer[sat_id] = {k: v.cpu() for k, v in current_local_wrapper.model_state_dict.items()}

                    # 상태 변경: 제출했으므로 다시 대기 상태 (다음 라운드 기다림)
                    sat_status[sat_id] = 'IDLE'

                    # 버퍼가 꽉 찼는지 확인 (Aggregation Trigger)
                    if len(agg_buffer) >= MIN_PARTICIPANTS:
                        self.sim_logger.info(f"\n⚡ [Sync Aggregation] {len(agg_buffer)} models collected! Updating Global Model v{int(global_version)} -> v{int(global_version)+1}")
                        
                        # 1. FedAvg (평균) - 직접 평균 계산
                        new_global_state = self.global_model_wrapper.model_state_dict.copy()
                        for key in new_global_state.keys():
                            if new_global_state[key].dtype == torch.float32:
                                # 버퍼에 있는 모든 모델의 해당 파라미터 스택
                                stack = torch.stack([m[key] for m in agg_buffer.values()])
                                # 평균 계산 (dim=0)
                                new_global_state[key] = torch.mean(stack, dim=0)
                        
                        # 2. 글로벌 모델 업데이트
                        new_version = global_version + 1.0
                        self.global_model_wrapper = PyTorchModel(
                            version=new_version,
                            model_state_dict=new_global_state,
                            trained_by=list(agg_buffer.keys())
                        )
                        self.global_model_net.load_state_dict(new_global_state)
                        
                        # 3. 평가 및 저장
                        g_acc, g_loss = self._evaluate_direct(
                            self.global_model_net, self.val_loader, sat_id="GS", version=new_version, stage="GLOBAL_TEST"
                        )
                        
                        if g_acc > self.best_acc:
                            self.best_acc = g_acc
                            save_dir = Path("./checkpoints")
                            save_dir.mkdir(parents=True, exist_ok=True)
                            
                            filename = f"sync_global_v{int(new_version)}_acc{g_acc:.2f}.pth"
                            save_path = save_dir / filename
                            
                            checkpoint = {
                                'model_state_dict': new_global_state,
                                'version': new_version,
                                'accuracy': g_acc,
                                'timestamp': datetime.now().isoformat(),
                                'round': new_version
                            }
                            torch.save(checkpoint, save_path)
                            self.sim_logger.info(f"   💾 [Save] New Best Model! ({self.best_acc:.2f}%)")
                        
                        self.sim_logger.info(f"   📢 Global Round {int(new_version)} Finished. Acc: {g_acc:.2f}%\n")
                        
                        # 4. 버퍼 초기화 (다음 라운드 시작)
                        agg_buffer.clear()

                # Case B: 글로벌 모델이 더 최신임 -> 다운로드 (Download)
                # 방금 Aggregation이 일어나서 버전이 올랐거나, 아직 구버전인 경우
                if self.global_model_wrapper.version > current_local_wrapper.version:
                    current_local_wrapper = PyTorchModel.from_model(
                        self.global_model_net, version=self.global_model_wrapper.version
                    )
                    self.satellite_models[sat_id] = current_local_wrapper
                    sat_status[sat_id] = 'IDLE' # 다운로드 받았으니 이제 학습 준비 완료
                    self.sim_logger.info(f"   📥 [SAT_{sat_id}] Downloaded Global v{self.global_model_wrapper.version:.0f}")

            # -----------------------------------------------------------
            # [이벤트 2] IoT 데이터 학습 (Local Training)
            # -----------------------------------------------------------
            elif event['type'] == 'IOT_TRAIN':
                # 조건: 현재 글로벌 모델과 버전이 같고, 아직 학습하지 않은 상태여야 함
                # (동기식이므로 한 라운드에 한 번만 학습)
                if current_local_wrapper.version == self.global_model_wrapper.version and sat_status[sat_id] == 'IDLE':
                    
                    self.sim_logger.info(f"\n📡 [Time: {event['start_time'].strftime('%m-%d %H:%M')}] SAT_{sat_id} Local Training (Round {int(current_local_wrapper.version)})")
                    
                    # 학습 설정
                    epochs = 5  # 로컬 에포크 (CIFAR-10 동기식은 5~10회 추천)
                    loader_idx = sat_id % len(self.client_subsets)
                    dataset = self.client_subsets[loader_idx]
                    
                    train_loader = DataLoader(
                        dataset, 
                        batch_size=128, 
                        shuffle=True, 
                        num_workers=8,
                        pin_memory=True
                    )
                    
                    current_local_wrapper.to_device(temp_model, device='cpu')
                    
                    # 학습 (ResNet9 Scratch이므로 LR 0.005 사용)
                    train_model(
                        model=temp_model,
                        global_state_dict=self.global_model_wrapper.model_state_dict,
                        train_loader=train_loader,
                        epochs=epochs,
                        lr=0.005,
                        device=self.device,
                        sim_logger=None # 로그가 너무 많으면 None, 보고 싶으면 self.sim_logger
                    )
                    
                    # 성능 기록
                    acc, _ = self._evaluate_direct(
                        temp_model, self.val_loader, sat_id, current_local_wrapper.version, "LOCAL_TRAIN"
                    )
                    self.satellite_performances[sat_id] = acc
                    
                    # 모델 상태 저장 (버전은 그대로, 상태만 업데이트)
                    current_local_wrapper = PyTorchModel.from_model(temp_model, version=current_local_wrapper.version)
                    self.satellite_models[sat_id] = current_local_wrapper
                    
                    sat_status[sat_id] = 'TRAINED' # 이제 지상국 만나면 업로드할 준비 완료
                    self.sim_logger.info(f"   ✅ Trained (Acc: {acc:.2f}%). Ready to Upload.")

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
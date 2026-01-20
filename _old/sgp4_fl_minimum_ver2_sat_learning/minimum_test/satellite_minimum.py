# minimum_test/satellite_minimum.py
import asyncio

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from datetime import datetime
from typing import Tuple, Dict
from ml.model import PyTorchModel, create_mobilenet
from ml.training import evaluate_model
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import KST
from config import LOCAL_EPOCHS, FEDPROX_MU
from simulation.clock import SimulationClock

# ----- CLASS DEFINITION ----- #
class Satellite:
    def __init__ (self, sat_id: int, satellite_obj: EarthSatellite, clock: 'SimulationClock', sim_logger, perf_logger, 
                  initial_model: PyTorchModel, train_loader, val_loader):
        self.sat_id = sat_id
        self.satellite_obj = satellite_obj
        self.clock = clock
        self.logger = sim_logger
        self.perf_logger = perf_logger
        self.position = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.state = "IDLE"
        self.local_model = initial_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.global_model = initial_model
        self.model_ready_to_upload = False
        self.miou = 0.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"SAT {self.sat_id} 생성")

    def _train_and_eval(self) -> Tuple[Dict, float, float]:
        """
        실제 PyTorch 모델 학습을 수행하는 블로킹(동기) 함수.
        asyncio 이벤트 루프를 막지 않기 위해 별도의 스레드에서 실행됩니다.
        """
        try:
            loader_length = len(self.train_loader)
            self.logger.info(f"✅ DataLoader의 총 배치 개수: {loader_length}")
            if loader_length == 0:
                self.logger.error("⚠️ DataLoader가 비어있습니다. Dataset을 확인해주세요.")
                return # 또는 다른 에러 처리
        except Exception as e:
            self.logger.error(f"❌ DataLoader의 길이를 확인하는 중 에러 발생: {e}")

        # --- 학습 파트 ---
        temp_model = create_mobilenet()
        temp_model.load_state_dict(self.local_model.model_state_dict)
        temp_model.to(self.device)
        temp_model.train()

        # --- FedProx 추가 부분 ---
        #    global_model_ref (w^t): Proximal term 계산을 위한 '고정된' 기준 모델
        #    마찬가지로 'self.global_model' (w^t)의 가중치를 가지며, 학습되지 않도록 .eval()
        global_model_ref = create_mobilenet()
        global_model_ref.load_state_dict(self.global_model.model_state_dict)
        global_model_ref.to(self.device)
        global_model_ref.eval() # 중요: gradient가 흐르지 않도록 설정

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.0003, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
        for epoch in range(LOCAL_EPOCHS):
            self.logger.info(f"    - SAT {self.sat_id}: 에포크 {epoch+1}/{LOCAL_EPOCHS} 진행 중...")
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = temp_model(images)
                loss = criterion(outputs, labels)
                
                # --- FedProx 손실 함수 수정 부분 ---
                #     근접 항(Proximal Term) 계산: ||w - w^t||^2
                prox_term = 0.0

                # temp_model.parameters() (w)와 global_model_ref.parameters() (w^t) 비교
                for local_param, global_param in zip(temp_model.parameters(), global_model_ref.parameters()):
                    # .detach()를 사용하여 w^t의 gradient가 계산되지 않도록 함
                    prox_term += torch.sum(torch.pow(local_param - global_param.detach(), 2))

                # --- FedProx 손실 함수 최종 계산 부분 ---
                #     최종 손실 계산: Loss + (mu/2) * prox_term
                total_loss = loss + (FEDPROX_MU / 2) * prox_term

                # loss.backward()
                total_loss.backward()
                optimizer.step()
            scheduler.step()
            
        new_state_dict = temp_model.cpu().state_dict()
        self.logger.info(f"  🧠 SAT {self.sat_id}: 로컬 학습 완료 ({LOCAL_EPOCHS} 에포크). 검증 시작...")
            
        # --- 검증 파트 ---
        accuracy, loss, miou = evaluate_model(new_state_dict, self.val_loader, self.device)
            
        return new_state_dict, accuracy, loss, miou

    async def train_and_eval(self):
        """CIFAR10 데이터셋으로 로컬 모델을 학습하고 검증"""
        self.state = 'TRAINING'
        self.logger.info(f"  ✅ SAT {self.sat_id}: 로컬 학습 시작 (v{self.local_model.version}).")
        new_state_dict = None
        try:
            # 현재 실행중인 이벤트 루프를 가져옵니다.
            loop = asyncio.get_running_loop()
            new_state_dict, accuracy, loss, miou = await loop.run_in_executor(None, self._train_and_eval)
            self.local_model.model_state_dict = new_state_dict
            self.miou = miou
            self.logger.info(f"  📊 [Local Validation] SAT: {self.sat_id}, Version: {self.local_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}, Miou: {miou:.2f}%")
            self.perf_logger.info(f"{datetime.now(KST).isoformat()},LOCAL_VALIDATION,{self.sat_id},{self.local_model.version},N/A,{accuracy:.4f},{loss:.6f},{miou:.4f}")

            self.local_model.trained_by = [self.sat_id]
            self.model_ready_to_upload = True

        except Exception as e:
            self.logger.error(f"  💀 SAT {self.sat_id}: 학습 또는 검증 중 에러 발생 - {e}", exc_info=True)

        finally:
            # 성공하든 실패하든 상태를 IDLE로 되돌립니다.
            self.state = 'IDLE'
            self.logger.info(f"  🏁 SAT {self.sat_id}: 학습 절차 완료.")

    async def send_model_to_iot(self, iot: 'IoT'):
        if self.global_model.version > iot.global_model.version:
            self.logger.info(f"  🛰️ SAT {self.sat_id} -> IoT {iot.name}: 글로벌 모델 전송 (버전 {self.global_model.version})")
            await iot.receive_global_model(self.global_model)

    async def receive_global_model(self, model: PyTorchModel):
        """지상국으로부터 글로벌 모델을 수신"""
        self.logger.info(f"  🛰️ SAT {self.sat_id}: 새로운 글로벌 모델 수신 (v{model.version}).")
        self.global_model = model
        self.local_model = model
        self.model_ready_to_upload = False

    async def send_local_model(self) -> PyTorchModel | None:
        if self.model_ready_to_upload:
            self.model_ready_to_upload = False
            return self.local_model
        return None
    
class Satellite_Manager:
    def __init__ (self, satellites: Dict[int, 'Satellite'], clock: 'SimulationClock', sim_logger):
        self.satellites = satellites
        self.logger = sim_logger
        self.clock = clock
        self.logger.info("위성 관리자 생성 완료.")

    async def run(self):
        self.logger.info("위성 관리자 운영 시작.")
        # while True:
        for sat_id, _ in self.satellites.items():
            self.logger.info(f"SAT {sat_id} 임무 시작.")
        await self.propagate_orbit()

    async def propagate_orbit(self):
        """시뮬레이션 시간에 맞춰 위성의 위치를 계속 업데이트"""
        while True:
            await asyncio.sleep(self.clock.real_interval)
            for _, sat in self.satellites.items():
                current_ts = self.clock.get_time_ts()
                geocentric = sat.satellite_obj.at(current_ts)
                subpoint = geocentric.subpoint()
                sat.position["lat"], sat.position["lon"], sat.position["alt"] = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km
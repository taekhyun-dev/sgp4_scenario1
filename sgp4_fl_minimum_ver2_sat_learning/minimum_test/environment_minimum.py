# minimum_test/environment_minimum.py
import asyncio
import torch
import os
from datetime import datetime
from skyfield.api import Topos
from typing import Dict
from ml.model import PyTorchModel
from ml.training import evaluate_model, weighted_update
from minimum_test.satellite_minimum import Satellite
from utils.logging_setup import KST
from config import AGGREGATION_STALENESS_THRESHOLD, IOT_FLYOVER_THRESHOLD_DEG
from simulation.clock import SimulationClock

# ----- CLASS DEFINITION ----- #
class IoT:
    def __init__ (self, name, latitude, longitude, elevation, sim_logger, initial_model: PyTorchModel, test_loader):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.global_model = initial_model
        self.test_loader = test_loader
        self.logger.info(f"IoT 클러스터 '{self.name}' 생성 완료.")

    async def run(self, clock: 'SimulationClock', satellites: Dict[int, 'Satellite']):
        self.logger.info(f"IoT 클러스터 '{self.name}' 운영 시작.")
        while True:
            current_ts = clock.get_time_ts()
            for sat_id, sat in satellites.items():
                elevation = (sat.satellite_obj - self.topos).at(current_ts).altaz()[0].degrees
                tasks = []
                if elevation >= IOT_FLYOVER_THRESHOLD_DEG:
                    self.logger.info(f"📡 [IoT 통신] IoT {self.name} <-> SAT {sat_id} 통신 시작 (고도각: {elevation:.2f}°)")
                    if sat.model_ready_to_upload:
                        # Local Model 수신 - I/O 작업이므로 코틀린
                        receive_model_task = asyncio.create_task(sat.send_model_to_iot(self))
                        tasks.append(receive_model_task)
                    # Local Update 진행 - CPU 작업이므로 프로세스 풀로 오프로딩
                    elif sat.state == 'IDLE' and not sat.model_ready_to_upload:
                        local_update_task = asyncio.create_task(sat.train_and_eval())
                        tasks.append(local_update_task)
                    await asyncio.gather(*tasks)
            await asyncio.sleep(clock.real_interval)
    
    async def receive_global_model(self, model: PyTorchModel):
        """위성으로부터 글로벌 모델을 수신"""
        if model.version > self.global_model.version:
            self.logger.info(f"  📡  IoT {self.name}: 새로운 글로벌 모델 수신 (v{model.version}).")
            self.global_model = model

class GroundStation:
    def __init__ (self, name, latitude, longitude, elevation, sim_logger, initial_model: PyTorchModel, test_loader, perf_logger, avg_data_count,
                   threshold_deg: float = 10.0, staleness_threshold: int = AGGREGATION_STALENESS_THRESHOLD):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.threshold_deg = threshold_deg
        self._comm_status: Dict[int, bool] = {}
        self.staleness_threshold = staleness_threshold
        self.global_model = initial_model
        self.test_loader = test_loader
        self.perf_logger = perf_logger
        self.best_miou = 0.0
        self.avg_data_count = avg_data_count
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"지상국 '{self.name}' 생성 완료. 글로벌 모델 버전: {self.global_model.version}")
        self.logger.info(f"  - Aggregation 정책: 버전 허용치 {self.staleness_threshold}")

    async def run(self, clock: 'SimulationClock', satellites: Dict[int, 'Satellite']):
        self.logger.info(f"지상국 '{self.name}' 운영 시작.")
        while True:
            current_ts = clock.get_time_ts()
            for sat_id, sat in satellites.items():
                elevation = (sat.satellite_obj - self.topos).at(current_ts).altaz()[0].degrees
                prev_visible = self._comm_status.get(sat_id, False)
                visible_now = elevation >= self.threshold_deg

                tasks = []
                # 통신 가능 시점
                if visible_now:
                    # AOS
                    if not prev_visible:
                        self.logger.info(f"📡 [AOS] {self.name} <-> SAT {sat_id} 통신 시작 (고도각: {elevation:.2f}°)")
                        sat.state = 'COMMUNICATING_GS'
                    # Local Model 수신
                    if sat.model_ready_to_upload:
                        receive_model_task = asyncio.create_task(self.receive_model_from_satellite(sat))
                        tasks.append(receive_model_task)
                    # Global Model 전송
                    if self.global_model.version > sat.local_model.version:
                        send_model_task = asyncio.create_task(self.send_model_to_satellite(sat))
                        tasks.append(send_model_task)
                # LOS
                elif prev_visible and not visible_now:
                    self.logger.info(f"📡 [LOS] {self.name} <-> SAT {sat_id} 통신 종료 (고도각: {elevation:.2f}°)")
                    sat.state = 'IDLE'
                self._comm_status[sat_id] = visible_now
                await asyncio.gather(*tasks)
            await asyncio.sleep(clock.real_interval)

    async def send_model_to_satellite(self, satellite: 'Satellite'):
        self.logger.info(f"  📤 {self.name} -> SAT {satellite.sat_id}: 글로벌 모델 전송 (버전 {self.global_model.version})")
        await satellite.receive_global_model(self.global_model)

    async def receive_model_from_satellite(self, satellite: 'Satellite'):
        local_model = await satellite.send_local_model()
        if local_model and self.global_model.version - local_model.version <= self.staleness_threshold:
            self.logger.info(f"  📥 {self.name} <- SAT {satellite.sat_id}: 로컬 모델 수신 완료 (버전 {local_model.version}, 학습자: {local_model.trained_by})")
            if satellite.miou < 50.0:  # 50% 미만은 아예 쳐다보지도 않음
                self.logger.warning(f"⚠️ Drop model from SAT {satellite.sat_id} (Miou: {satellite.miou:.2f}%)")
                return
            # 2. [신규/핵심] 상대적 성능 검사 (후반 방어용)
            # 글로벌 모델이 어느 정도 학습된 상태(예: mIoU 50 이상)라면 더 엄격하게 봄
            if self.best_miou > 50.0:
                # 글로벌 최고 기록의 70% 도 안되는 모델은 노이즈로 간주하고 폐기
                relative_threshold = self.best_miou * 0.7 
                
                if local_model.miou < relative_threshold:
                    self.logger.warning(f"🛡️ [Drop] SAT {satellite.sat_id} 성능 미달 (Local: {satellite.miou:.2f}% < Global Best의 70%: {relative_threshold:.2f}%)")
                    return
            # Local Model 수신 후 Aggregation 진행 - I/O 작업이므로 코틀린
            await self.try_aggregate_and_update(satellite, local_model)
        else:
             self.logger.warning(f"⚠️ [Drop] SAT {satellite.sat_id} 모델 폐기 (Too Stale: v{local_model.version} vs v{self.global_model.version})")
             return

    def calculate_mixing_weight(self, local_version, current_version, local_miou, local_data_count, avg_data_count):
        import numpy as np
        """
        Aggregation 가중치(alpha)를 동적으로 계산하는 함수 (연구 차별점)

        Args:
        local_data_count: 해당 위성이 학습에 사용한 배치 개수 (예: 212)
        avg_data_count: 클러스터 내 위성들의 평균 배치 개수 (기준값)
        """
        BASE_ALPHA = 0.1  # 기본 반영 비율 (보수적 접근)
        global_miou = self.best_miou
        
        # 1. Staleness 패널티
        # 버전 차이가 클수록 반영 비율이 1/2, 1/3... 로 줄어듦
        staleness = max(0, current_version - local_version)
        staleness_factor = 1.0 / (1.0 + staleness) 

        perf_ratio = 1.0
        
        # 2. Performance (성능) 가중치
        # 로컬 모델이 글로벌 모델보다 성능이 좋으면 더 적극적으로 반영 (최대 2배)
        # 성능이 나쁘면 반영 비율 감소 (최소 0.5배)
        if global_miou > 0:
            perf_ratio = local_miou / global_miou
            # perf_ratio를 0.5 ~ 2.0 사이로 클리핑하여 안정성 확보
            performance_factor = np.clip(perf_ratio, 0.5, 2.0)
        else:
            performance_factor = 1.0

        if avg_data_count > 0:
            data_ratio = local_data_count / avg_data_count
            # [전략] 데이터 많은 위성(SAT 4)을 살리기 위해 범위를 0.05 ~ 10.0으로 설정 (아주 좋음)
            data_factor = np.clip(data_ratio, 0.05, 10.0)
        else:
            data_ratio = 1.0
            data_factor = 1.0

        # data_ratio = local_data_count / avg_data_count
        # # data_factor = np.clip(data_ratio, 0.1, 5.0)
        # data_factor = np.clip(data_ratio, 0.05, 10.0)
        # 최종 반영 비율 계산 (보통 0.05 ~ 0.2 사이가 됨)
        # final_alpha = BASE_ALPHA * staleness_factor * performance_factor

        if perf_ratio > 1.0 or data_ratio > 2.0:
            staleness_factor = 1.0
        final_alpha = BASE_ALPHA * staleness_factor * performance_factor * data_factor

        # [수정] 글로벌 모델 성능에 따른 동적 제한
        if self.best_miou > 80.0:
            # 이미 80점 넘으면 최대 10%까지만 반영 (조심조심 튜닝)
            MAX_ALPHA_LIMIT = 0.1 
        elif self.best_miou > 60.0:
            MAX_ALPHA_LIMIT = 0.3
        else:
            # 초반에는 과감하게 50%까지 허용
            MAX_ALPHA_LIMIT = 0.5
            
        final_alpha = min(final_alpha, MAX_ALPHA_LIMIT)
        # final_alpha = min(final_alpha, 1.0)
        
        return final_alpha, staleness_factor, performance_factor, data_factor

    async def try_aggregate_and_update(self, sat: Satellite, local_model: PyTorchModel):
        """Aggregation 수행"""
        sat_id = sat.sat_id

        self.logger.info(f"✨ [{self.name} Aggregation] 진행 - SAT {sat_id}의 v{local_model.version} 로컬 모델과 기존 글로벌 모델(v{self.global_model.version}) 취합 시작...")
        
        current_global_miou = self.best_miou
        local_batch_count = len(sat.train_loader)

        # --- Dynamic Mixing Weight 계산 ---
        alpha, s_factor, p_factor, d_factor = self.calculate_mixing_weight(
            local_model.version, self.global_model.version, sat.miou, local_batch_count, self.avg_data_count
        )

        self.logger.info(f"✨ [{self.name} Aggregation] SAT {sat_id} 반영 시작")
        self.logger.info(f"    - Staleness: {s_factor:.2f} (Ver Diff: {self.global_model.version - local_model.version})")
        self.logger.info(f"    - Performance: {p_factor:.2f} (Local: {sat.miou:.2f}% / Global: {current_global_miou:.2f}%)")
        self.logger.info(f"    - Data Volume: {d_factor:.2f} (Local: {local_batch_count} / Avg: 36)")
        self.logger.info(f"   👉 최종 반영 비율(Alpha): {alpha:.4f}")

        new_state_dict = weighted_update(
            global_state_dict=self.global_model.model_state_dict, 
            local_state_dict=local_model.model_state_dict, 
            alpha=alpha, 
            device=self.device
        )

        # state_dicts_to_avg = [self.global_model.model_state_dict] + [local_model.model_state_dict]
        # new_state_dict = fed_avg(state_dicts_to_avg)
        
        new_version = self.global_model.version + 1 # 버전업
        all_contributors = list(set(self.global_model.trained_by + [p for p in local_model.trained_by]))
        self.global_model = PyTorchModel(version=new_version, model_state_dict=new_state_dict, trained_by=all_contributors)
        self.logger.info(f"✨ [{self.name} Aggregation] 새로운 글로벌 모델 생성 완료! (버전 {self.global_model.version})")

        # evaluate
        loop = asyncio.get_running_loop()
        accuracy, loss, miou = await loop.run_in_executor(None, evaluate_model, self.global_model.model_state_dict, self.test_loader, self.device)

        self.logger.info(f"  🧪 [Global Test] Owner: {self.name}, Version: {self.global_model.version}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}, Miou: {miou:.2f}%")
        self.perf_logger.info(f"{datetime.now(KST).isoformat()},GLOBAL_TEST,{self.name},{self.global_model.version},N/A,{accuracy:.4f},{loss:.6f},{miou:.4f}")
        if miou > self.best_miou:
            previous_best = self.best_miou
            self.best_miou = miou
            
            save_dir = "./checkpoints/global"
            os.makedirs(save_dir, exist_ok=True)
            
            # 파일명에 miou 포함
            save_path = os.path.join(save_dir, f"best_global_model_v{new_version}_miou{miou:.2f}.pth")
            
            await loop.run_in_executor(None, torch.save, self.global_model.model_state_dict, save_path)
            
            self.logger.info(f" 💾 [Save] New Best mIoU Model! ({previous_best:.2f}% -> {self.best_miou:.2f}%)")
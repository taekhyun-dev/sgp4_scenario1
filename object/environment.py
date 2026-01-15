# minimum_test/environment_minimum.py
import asyncio
import torch
import os
from datetime import datetime
from skyfield.api import Topos
from typing import Dict
from ml.model import PyTorchModel
from ml.training import evaluate_model, weighted_update
# from object.satellite import Satellite
from utils.logging_setup import KST
from config import AGGREGATION_STALENESS_THRESHOLD, IOT_FLYOVER_THRESHOLD_DEG
# from simulation.clock import SimulationClock

# ----- CLASS DEFINITION ----- #
class IoT:
    def __init__ (self, name, latitude, longitude, elevation, sim_logger, initial_model: PyTorchModel, test_loader):
        self.name = name
        self.logger = sim_logger
        self.topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude, elevation_m=elevation)
        self.global_model = initial_model
        self.test_loader = test_loader
        self.logger.info(f"IoT 클러스터 '{self.name}' 생성 완료.")
    
    async def run(self):
        self.logger.info(f"IoT 클러스터 '{self.name}' 운영 시작.")
        # IoT 클러스터 운영 로직 구현
        await asyncio.sleep(1)  # 예시: 1초 대기
        self.logger.info(f"IoT 클러스터 '{self.name}' 운영 완료.")

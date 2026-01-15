import asyncio
from typing import List, Dict, Coroutine
from datetime import datetime, timezone, timedelta
from utils.monitor import SystemMonitor
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import KST, setup_loggers
from pathlib import Path
from ml.data import get_cifar10_loaders
from ml.model import PyTorchModel, create_mobilenet
from config import NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, NUM_WORKERS
from object.satellite import Satellite_Manager

def load_constellation(tle_path: str, sim_logger) -> Dict[int, EarthSatellite]:
    """TLE 파일에서 위성군 정보를 불러오는 함수"""
    if not Path(tle_path).exists(): raise FileNotFoundError(f"'{tle_path}' 파일을 찾을 수 없습니다.")
    satellites = {}
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]; i = 0
        while i < len(lines):
            name, line1, line2 = lines[i:i+3]; sat_id = int(name.replace("SAT", ""))
            satellites[sat_id] = EarthSatellite(line1, line2, name)
            i += 3
    sim_logger.info(f"총 {len(satellites)}개의 위성을 TLE 파일에서 불러왔습니다.")
    return satellites

def create_simulation_environment(eval_infra: dict):
    sim_logger = eval_infra['sim_logger']
    start_time = eval_infra['start_time']
    end_time = eval_infra['end_time']

    # TLE 데이터 로드
    satellites = load_constellation("constellation.tle", sim_logger)

    # 초기 글로벌 모델 생성
    initial_pytorch_model = create_mobilenet()
    initial_global_model = PyTorchModel(version=0, model_state_dict=initial_pytorch_model.state_dict())

    sat_manager = Satellite_Manager(start_time, end_time, sim_logger, satellites)


async def main():
    try:
        sim_logger, perf_logger = setup_loggers()

        sim_logger.info("Loading CIFAR10 dataset...")
        avg_data_count, client_loaders, val_loader, test_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, dirichlet_alpha=DIRICHLET_ALPHA,
                                                                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        sim_logger.info("Dataset loaded.")

        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=24)

        current_time = start_time.strftime("%Y%m%d_%H%M%S")
        resource_monitor = SystemMonitor(
            logger=sim_logger, 
            interval=1.0, 
            log_file=f"./logs/resource_{current_time}.csv" # 로그 파일명 지정
        )
        monitor_task = asyncio.create_task(resource_monitor.run())

        # 로드된 데이터를 전달하여 시뮬레이션 환경 구성
        sim_logger.info("시뮬레이션 환경을 구성 ...")

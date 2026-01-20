'''
    IoT -> 위성으로 데이터를 전달
    위성에서 로컬학습 진행
    위성 -> 지상국 로컬 모델 전달
    지상국에서 글로벌 학습 진행
    지상국 -> 위성 새로운 모델 전달
    위성 -> IoT 새로운 모델 전달
'''
import torch

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict
from utils.skyfield_utils import EarthSatellite
from utils.logging_setup import setup_loggers
from ml.data import get_cifar10_loaders
from simulation.clock import SimulationClock

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

if __name__ == "__main__":
    sim_logger, perf_logger = setup_loggers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터셋 준비
    sim_logger.info("Loading CIFAR10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    sim_logger.info("Dataset loaded.")

    # TLE 데이터 로드
    all_sats_skyfield = load_constellation("constellation.tle", sim_logger)

    # 시뮬레이션 시간 설정
    start_time = datetime.now(timezone.utc)
    simulation_clock = SimulationClock(
        start_dt=start_time, 
        time_step=timedelta(minutes=10),
        real_interval=1.0,
        sim_logger=sim_logger
    )
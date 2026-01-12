import torch
from minimum_test.environment_minimum import IoT, GroundStation
from simulation.clock import SimulationClock
from typing import Dict
from pathlib import Path
from utils.skyfield_utils import EarthSatellite

def load_satellites_from_tle(tle_path: str, sim_logger) -> Dict[int, EarthSatellite]:
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

class SimulationManager:
    def __init__(self):
        # 1. 초기화: 모든 객체를 여기서 생성하고 self에 저장
        self.clock = SimulationClock(...)
        self.ground_station = GroundStation(...)
        self.satellites = load_satellites_from_tle(...)
        self.iot_clusters = [...]
        
    def step(self):
        # 2. 1초(또는 1스텝) 진행하는 로직
        self.clock.tick()
        check_aos(self.ground_station, self.satellites)
        # ... 기존 while 루프 안의 내용들 ...

    def run(self):
        # 3. 메인 루프 실행
        while self.clock.current_time < END_TIME:
            self.step()

    # ★ 핵심: 이 클래스가 모든 정보를 다 가지고 있으므로 저장이 쉬워짐
    def save_checkpoint(self, path):
        checkpoint = {
            'time': self.clock.current_time,
            'gs': self.ground_station, 
            'sats': self.satellites,
            # ...
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        # ... (이전 답변의 load 로직)
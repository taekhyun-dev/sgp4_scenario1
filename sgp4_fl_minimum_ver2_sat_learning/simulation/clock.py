# simulation/clock.py
import asyncio
from datetime import datetime, timedelta
from utils.skyfield_utils import to_ts
from utils.logging_setup import KST

class SimulationClock:
    """
    시뮬레이션의 전역 시간을 관리하는 클래스.
    '실제 시간'과 '시뮬레이션 속의 가상 시간'을 동기화합니다.
    """
    def __init__(self, start_dt: datetime, time_step: timedelta, real_interval: float, sim_logger=None):
        self._current_dt = start_dt
        self.time_step = time_step
        self.real_interval = real_interval
        self.logger = sim_logger
        speed = time_step.total_seconds() / real_interval
        self.logger.info(f"시뮬레이션 시계 생성. 시작: {start_dt.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S %Z')}, 1초당 {time_step.total_seconds()}초 진행 (x{speed:.0f} 배속)")

    async def run(self):
        """real_interval만큼 대기 후, time_step만큼 시뮬레이션 시간을 진행시키는 루프"""
        while True:
            self._current_dt += self.time_step
            await asyncio.sleep(self.real_interval)

    def get_time_datetime(self) -> datetime:
        """현재 시뮬레이션 시간을 datetime 객체로 반환"""
        return self._current_dt
        
    def get_time_ts(self):
        """현재 시뮬레이션 시간을 Skyfield의 Time 객체로 반환"""
        return to_ts(self._current_dt)

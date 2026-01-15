from datetime import datetime, timedelta
from utils.skyfield_utils import EarthSatellite
from typing import List, Tuple, Dict
from pathlib import Path
from config import IOT_FLYOVER_THRESHOLD_DEG, GS_FLYOVER_THRESHOLD_DEG
import numpy as np

class Satellite_Manager:
    def __init__(self, start_time: datetime, end_time: datetime):
        self.start_time = start_time
        self.end_time = end_time
        # self.logger = sim_logger
        self.satellites:Dict[int, EarthSatellite] = {}
        self.satellites_positions:Dict[int, List[Tuple[float, float, float]]] = {}
        # self.logger.info("위성 관리자 생성 완료.")
        print("위성 관리자 생성 완료.")

    def date_range(self, start, end, step):
        curr = start
        while curr <= end:
            yield curr
            curr += step

    # -------- TEST -------- #
    def load_constellation(self) -> Dict[int, EarthSatellite]:
        """TLE 파일에서 위성군 정보를 불러오는 함수"""
        tle_path = "constellation.tle"
        if not Path(tle_path).exists(): raise FileNotFoundError(f"'{tle_path}' 파일을 찾을 수 없습니다.")
        satellites = {}
        with open(tle_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]; i = 0
            while i < len(lines):
                name, line1, line2 = lines[i:i+3]; sat_id = int(name.replace("SAT", ""))
                satellites[sat_id] = EarthSatellite(line1, line2, name)
                i += 3

        self.satellites = satellites
    # -------- TEST -------- #

    async def run(self):
        # self.logger.info("위성 관리자 운영 시작.")
        print("위성 관리자 운영 시작.")
        step = timedelta(days=1)
        self.load_constellation()

        for i in range(0, 50):
            self.satellites_positions[i] = []

        for t in self.date_range(self.start_time, self.end_time, step):
            await self.propagate_orbit(t, t + step)

    async def propagate_orbit(self, start_time, end_time):
        step = timedelta(seconds=10)
        self.times = []
        curr = start_time
        while curr < end_time:
            self.times.append(curr)
            curr += step

        from skyfield.api import load
        ts = load.timescale() 
        self.t_vector = ts.from_datetimes(self.times)

        for sat_id, satellite in self.satellites.items():
            geocentric = satellite.at(self.t_vector)
            subpoint = geocentric.subpoint()

            lats = subpoint.latitude.degrees
            lons = subpoint.longitude.degrees
            elevs = subpoint.elevation.km

            trajectory = list(zip(lats, lons, elevs))

            self.satellites_positions[sat_id].extend(trajectory)
        # for sat_id, satellite in self.satellites.items():
        #     for t in self.date_range(start_time, end_time, step):
        #         time = to_ts(t)
        #         geocentric = satellite.at(time)
        #         subpoint = geocentric.subpoint()
        #         self.satellites_positions[sat_id].append((subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km))
        
        # self.logger.info(f"{start_time}의 위성 위치 업데이트 완료.")
        print(f"{start_time}의 위성 위치 업데이트 완료.")
        await self.check_iot_comm()
        print(f"{start_time}의 IoT 지상국 통신 가능 시간 분석 완료.")
        await self.check_gs_comm()
        print(f"{start_time}의 지상국 통신 가능 시간 분석 완료.")

    async def check_iot_comm(self):
        print("IoT 지상국 통신 가능 시간 분석 시작...")

        from skyfield.api import wgs84
        iot_devices = [
            {"name": "Amazon_Forest", "loc": wgs84.latlon(-3.47, -62.37, elevation_m=100)},
            {"name": "Great_Barrier_Reef", "loc": wgs84.latlon(-18.29, 147.77, elevation_m=0)},
            {"name": "Abisko Tundra", "loc": wgs84.latlon(68.35, 18.79, elevation_m=420)},
        ]     

        for iot in iot_devices:
            print(f"--- Analyzing {iot['name']} ---")
            for sat_id, satellite in self.satellites.items():
                difference = satellite - iot['loc']
                topocentric = difference.at(self.t_vector)
                alt, az, distance = topocentric.altaz()
                visible_indices = np.where(alt.degrees > IOT_FLYOVER_THRESHOLD_DEG)[0]

                if len(visible_indices) == 0:
                    continue

                breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
                windows = np.split(visible_indices, breaks)

                print(f"📡 [SAT_{sat_id} <-> {iot['name']}] 총 {len(windows)}회 접속 발생")

                for i, window in enumerate(windows):
                    start_idx = window[0]
                    end_idx = window[-1]
                    
                    start_time = self.times[start_idx]
                    end_time = self.times[end_idx]
                    duration = end_time - start_time
                    
                    # 10초 미만(점 하나)인 경우 duration이 0으로 나올 수 있으므로 보정 (선택사항)
                    if duration.total_seconds() == 0:
                        duration = timedelta(seconds=10)

                    print(f"  └─ #{i+1}: {start_time.strftime('%H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')} ({duration} 유지)")

    async def check_gs_comm(self):
        print("IoT 지상국 통신 가능 시간 분석 시작...")

        from skyfield.api import wgs84
        gs = {"name": "Ground Station", "loc": wgs84.latlon(37.5665, 126.9780, elevation_m=34)}

        print(f"--- Analyzing {gs['name']} ---")

        for sat_id, satellite in self.satellites.items():
            difference = satellite - gs['loc']
            topocentric = difference.at(self.t_vector)
            alt, az, distance = topocentric.altaz()
            visible_indices = np.where(alt.degrees > GS_FLYOVER_THRESHOLD_DEG)[0]

            if len(visible_indices) == 0:
                continue

            breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
            windows = np.split(visible_indices, breaks)

            print(f"📡 [SAT_{sat_id} <-> {gs['name']}] 총 {len(windows)}회 접속 발생")

            for i, window in enumerate(windows):
                start_idx = window[0]
                end_idx = window[-1]
                
                start_time = self.times[start_idx]
                end_time = self.times[end_idx]
                duration = end_time - start_time
                
                # 10초 미만(점 하나)인 경우 duration이 0으로 나올 수 있으므로 보정 (선택사항)
                if duration.total_seconds() == 0:
                    duration = timedelta(seconds=10)

                print(f"  └─ #{i+1}: {start_time.strftime('%H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')} ({duration} 유지)")

from datetime import datetime, timezone, timedelta

start_time = datetime.now(timezone.utc)
sat_manager = Satellite_Manager(start_time, start_time + timedelta(days=1))
import asyncio
asyncio.run(sat_manager.run())
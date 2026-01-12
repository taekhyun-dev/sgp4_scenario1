# utils/skyfield_utils.py
from datetime import datetime
from skyfield.api import load, EarthSatellite as SkyfieldEarthSatellite

# --- Skyfield 및 시간 관련 헬퍼 함수 ---
_ts = load.timescale()

class EarthSatellite(SkyfieldEarthSatellite):
    """Skyfield의 EarthSatellite 객체를 래핑하여 timescale을 내부적으로 관리"""
    def __init__(self, line1, line2, name):
        super().__init__(line1, line2, name, _ts)

def to_ts(dt: datetime):
    """datetime 객체를 Skyfield의 Time 객체로 변환"""
    return _ts.from_datetime(dt)

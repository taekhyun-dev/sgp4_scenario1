# utils/logging_setup.py
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- 로깅 설정 (KST 적용) ---
KST = timezone(timedelta(hours=9))

class KSTFormatter(logging.Formatter):
    """로그 시간을 한국 표준시(KST)로 변환하는 포매터"""
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, KST)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s

def setup_loggers():
    """시뮬레이션 및 성능 로그를 설정하고 로거 객체를 반환"""
    # 로그 파일 이름에 타임스탬프 추가 (KST 기준)
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # --- 일반 시뮬레이션 로거 설정 ---
    sim_logger = logging.getLogger("simulation")
    # 핸들러 중복 추가 방지
    if sim_logger.hasHandlers():
        sim_logger.handlers.clear()
        
    sim_logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    sim_handler = logging.FileHandler(log_dir / f"simulation_{timestamp}.log", mode='w')
    sim_formatter = KSTFormatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    sim_handler.setFormatter(sim_formatter)
    sim_logger.addHandler(sim_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(sim_formatter)
    sim_logger.addHandler(console_handler)

    # --- 성능 평가 결과 로거 (CSV) 설정 ---
    perf_logger = logging.getLogger("performance")
    if perf_logger.hasHandlers():
        perf_logger.handlers.clear()
        
    perf_logger.setLevel(logging.INFO)

    perf_handler = logging.FileHandler(log_dir / f"performance_{timestamp}.csv", mode='w')
    perf_formatter = KSTFormatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    
    # CSV 헤더 작성
    perf_handler.stream.write("timestamp,event_type,owner_id,model_version,cluster_version,accuracy,loss,miou\n")
    perf_handler.setFormatter(perf_formatter)
    perf_logger.addHandler(perf_handler)
    
    # 전파 방지
    sim_logger.propagate = False
    perf_logger.propagate = False
    
    return sim_logger, perf_logger

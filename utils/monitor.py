import asyncio
import psutil
import os
from datetime import datetime
import torch

# GPU 모니터링 라이브러리 (없으면 에러 방지)
try:
    import pynvml
    HAS_GPU_MONITOR = True
except ImportError:
    HAS_GPU_MONITOR = False

class SystemMonitor:
    def __init__(self, logger, interval=1.0, log_file="resource_usage.csv"):
        """
        Args:
            logger: 일반 로거 (콘솔 출력용)
            interval: 모니터링 주기 (초)
            log_file: CSV로 저장할 파일 경로
        """
        self.logger = logger
        self.interval = interval
        self.running = False
        self.process = psutil.Process(os.getpid()) # 현재 시뮬레이션 프로세스 ID
        self.log_file = log_file
        
        # GPU 초기화
        self.gpu_handle = None
        if HAS_GPU_MONITOR and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                # 0번 GPU 사용 가정 (멀티 GPU면 수정 필요)
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) 
                self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                self.logger.info(f"🖥️ GPU Monitor Initialized: {self.gpu_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ GPU monitoring failed: {e}")

        # CSV 헤더 작성
        self._init_csv()

    def _init_csv(self):
        # 파일이 없으면 헤더 생성
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                header = "timestamp,cpu_percent,memory_mb,gpu_util,gpu_mem_mb\n"
                f.write(header)

    def get_stats(self):
        # 1. CPU & Memory (현재 프로세스 기준)
        # cpu_percent(interval=None)은 비차단(non-blocking) 호출
        cpu_usage = self.process.cpu_percent(interval=None) / psutil.cpu_count()
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024) # Byte -> MB 변환

        # 2. GPU Stats
        gpu_util = 0.0
        gpu_mem = 0.0
        
        if self.gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                
                gpu_util = util.gpu # GPU 코어 사용률 (%)
                gpu_mem = mem.used / (1024 * 1024) # VRAM 사용량 (MB)
            except Exception:
                pass
        
        return cpu_usage, mem_mb, gpu_util, gpu_mem

    async def run(self):
        self.running = True
        self.logger.info(f"📊 System Resource Monitor Started (Interval: {self.interval}s)")
        
        # CPU 초기화 (첫 호출은 0.0일 수 있음)
        self.process.cpu_percent(interval=None)

        while self.running:
            try:
                # 데이터 수집
                cpu, ram, gpu_util, gpu_mem = self.get_stats()
                
                # 타임스탬프
                now = datetime.now().isoformat()
                
                # CSV 기록
                with open(self.log_file, "a") as f:
                    f.write(f"{now},{cpu:.2f},{ram:.2f},{gpu_util:.2f},{gpu_mem:.2f}\n")
                
                # (선택) 로그 레벨이 DEBUG면 콘솔에도 출력
                # self.logger.debug(f"[Res] CPU: {cpu:.1f}% | RAM: {ram:.0f}MB | GPU: {gpu_util}% | VRAM: {gpu_mem:.0f}MB")

            except Exception as e:
                self.logger.error(f"❌ Monitor Error: {e}")

            # 주기 대기
            await asyncio.sleep(self.interval)

    def stop(self):
        self.running = False
        if self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
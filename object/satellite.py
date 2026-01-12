from datetime import datetime, timedelta

class Satellite_Manager:
    def __init__(self, start_time: datetime, end_time: datetime, sim_logger):
        self.start_time = start_time
        self.end_time = end_time
        self.logger = sim_logger
        self.logger.info("위성 관리자 생성 완료.")

    async def run(self):
        self.logger.info("위성 관리자 운영 시작.")
        await self.propagate_orbit()

    async def propagate_orbit(self):
        # 2. 시작/종료 시간 및 간격 설정
        # 간격 설정 (예: 30분 간격)
        step = timedelta(minutes=10) 

        # 3. 반복문 실행
        curr_time = self.start_time

        while curr_time <= self.end_time:
            # 함수에 시간 인자 전달
            process_data(curr_time)
            
            # 시간 증가
            curr_time += step
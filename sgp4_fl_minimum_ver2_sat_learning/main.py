# main.py
import asyncio
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Coroutine
from datetime import datetime, timezone, timedelta
from utils.logging_setup import KST, setup_loggers
from utils.monitor import SystemMonitor
from utils.skyfield_utils import EarthSatellite
from ml.data import get_cifar10_loaders
from simulation.clock import SimulationClock
from minimum_test.satellite_minimum import Satellite, Satellite_Manager
from ml.model import PyTorchModel, create_mobilenet
from minimum_test.environment_minimum import IoT, GroundStation
from config import NUM_CLIENTS, DIRICHLET_ALPHA, BATCH_SIZE, NUM_WORKERS

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

def create_simulation_environment(
    clock: 'SimulationClock', 
    eval_infra: dict
):
    """
    시뮬레이션 환경을 구성하는 모든 객체(위성, 지상국, IoT)를 생성합니다.
    main.py에서 TLE 로딩 후, 이 함수를 호출하여 환경을 구성합니다.
    """
    sim_logger = eval_infra['sim_logger']
    perf_logger = eval_infra['perf_logger']
    client_loaders = eval_infra['client_loaders']
    avg_data_count = eval_infra['avg_data_count']
    val_loader = eval_infra['val_loader']
    test_loader = eval_infra['test_loader']

    # TLE 데이터 로드
    all_sats_skyfield = load_constellation("constellation.tle", sim_logger)
    
    # 초기 글로벌 모델 생성
    initial_pytorch_model = create_mobilenet()
    initial_global_model = PyTorchModel(version=0, model_state_dict=initial_pytorch_model.state_dict())
    
    # 지상국 및 IoT 클러스터 생성
    ground_stations = [
        GroundStation("Seoul-GS", 37.5665, 126.9780, 34, sim_logger, initial_model=initial_global_model, test_loader=test_loader,
                      perf_logger=perf_logger, avg_data_count=avg_data_count),
        # GroundStation("Houston-GS", 29.7604, -95.3698, 12, initial_model=initial_global_model, eval_infra=eval_infra)
    ]
    
    iot_clusters = [
        IoT("Amazon_Forest", -3.47, -62.37, 100, sim_logger, initial_model = initial_global_model, test_loader=test_loader),
        IoT("Great_Barrier_Reef", -18.29, 147.77, 0, sim_logger, initial_model = initial_global_model, test_loader=test_loader),
        IoT("Siberian_Tundra", 68.35, 18.79, 420, sim_logger, initial_model = initial_global_model, test_loader=test_loader)
    ]

    # 3. 위성 객체 및 클러스터 구성 (기존 main.py 로직)
    satellites_in_sim: Dict[int, Satellite] = {}
    sat_ids = sorted(list(all_sats_skyfield.keys()))

    for sat_id in sat_ids:
        train_loader = client_loaders[sat_id]
        sat = Satellite(
            sat_id, all_sats_skyfield[sat_id], clock, sim_logger, perf_logger,
            initial_global_model, train_loader, val_loader
        )
        satellites_in_sim[sat_id] = sat
            
    sim_logger.info(f"총 {len(satellites_in_sim)}개 위성 생성 완료.")

    sat_manager = Satellite_Manager(satellites_in_sim, clock, sim_logger)
    
    return sat_manager, satellites_in_sim, ground_stations, iot_clusters

async def main():
    try:
        sim_logger, perf_logger = setup_loggers()

        sim_logger.info("Loading CIFAR10 dataset...")
        avg_data_count, client_loaders, val_loader, test_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, dirichlet_alpha=DIRICHLET_ALPHA,
                                                                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        sim_logger.info("Dataset loaded.")

        eval_infra = {
            "sim_logger": sim_logger,
            "perf_logger": perf_logger,
            "avg_data_count": avg_data_count,
            "client_loaders": client_loaders,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }

        start_time = datetime.now(timezone.utc)
        simulation_clock = SimulationClock(
            start_dt=start_time, 
            time_step=timedelta(seconds=10),
            real_interval=1.0,
            sim_logger=sim_logger
        )

        current_time = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        resource_monitor = SystemMonitor(
            logger=sim_logger, 
            interval=1.0, 
            log_file=f"./logs/resource_{current_time}.csv" # 로그 파일명 지정
        )
        monitor_task = asyncio.create_task(resource_monitor.run())

        # 로드된 데이터를 전달하여 시뮬레이션 환경 구성
        sim_logger.info("시뮬레이션 환경을 구성 ...")
        sat_manager, satellites, ground_stations, iot_clusters = create_simulation_environment(
            simulation_clock, eval_infra
        )
        # satellites 리스트 출력
        print("Loaded Satellites:")
        for _, sat in satellites.items():
            print(f"  - SAT ID: {sat.sat_id}, Position: {sat.position}, State: {sat.state}")

        sim_logger.info("환경 구성 완료.")

        # 시뮬레이션 메인 태스크들
        sim_tasks: List[Coroutine] = [
            simulation_clock.run(),
            *[gs.run(simulation_clock, satellites) for gs in ground_stations],
            *[iot.run(simulation_clock, satellites) for iot in iot_clusters],
            sat_manager.run()
        ]
        sim_logger.info("시뮬레이션을 시작합니다.")
        try:
            await asyncio.gather(*sim_tasks)
        except KeyboardInterrupt:
            sim_logger.info("Simulation Interrupted.")
        finally:
            # 4. 종료 시 모니터 정리
            resource_monitor.stop()
            await monitor_task
            sim_logger.info("Simulation Finished.")
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        # 예기치 않은 에러 발생 시 로깅
        sim_logger, _ = setup_loggers()
        sim_logger.error(f"\n시뮬레이션 중 치명적인 에러 발생: {e}", exc_info=True)
    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    asyncio.run(main())
    
import torch
import torch.multiprocessing as mp
import time
import os

# 각 작업자 프로세스가 실행할 함수
# tasks_queue에서 작업을 가져와 연산 후 results_queue에 결과를 넣습니다.
def worker(worker_id, tasks_queue, results_queue):
    print(f"작업자 프로세스 {worker_id} (PID: {os.getpid()}) 시작")
    while True:
        task = tasks_queue.get()
        # 'None'을 받으면 작업을 종료하라는 신호로 인식
        if task is None:
            break
        
        # 실제 작업 수행 (예: 텐서에 2를 곱하고 1을 더함)
        result = task * 2 + 1
        time.sleep(0.5) # 실제 작업이 시간이 걸리는 것을 시뮬레이션
        results_queue.put(result)
    print(f"작업자 프로세스 {worker_id} 종료")


# 메인 스크립트 실행 보호
if __name__ == "__main__":
    # torch.multiprocessing을 사용할 때 권장되는 시작 방식
    mp.set_start_method('spawn', force=True)

    # 프로세스 간 통신을 위한 큐 생성
    tasks = mp.Queue()
    results = mp.Queue()

    num_processes = 4  # 사용할 프로세스 수
    processes = []

    # 작업자 프로세스 생성 및 시작
    for i in range(num_processes):
        p = mp.Process(target=worker, args=(i, tasks, results))
        p.start()
        processes.append(p)

    # 메인 프로세스에서 작업 생성 및 큐에 추가
    print(f"메인 프로세스 (PID: {os.getpid()})가 작업을 생성합니다.")
    for i in range(10):
        data = torch.randn(2, 2) # 2x2 랜덤 텐서 생성
        tasks.put(data)

    # 모든 작업이 끝났음을 작업자들에게 알리기 위해 'None'을 추가
    for _ in range(num_processes):
        tasks.put(None)

    # 결과 수집
    for i in range(10):
        res = results.get()
        print(f"{i+1}번째 결과 수신:\n{res}")

    # 모든 작업자 프로세스가 종료될 때까지 대기
    for p in processes:
        p.join()
    
    print("모든 작업 완료!")
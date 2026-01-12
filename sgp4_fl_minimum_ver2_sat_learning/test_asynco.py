import asyncio
import time
import random

# GPU Worker: GPU에서 실제 모델이 학습되는 과정을 모사
async def gpu_worker(name: str, queue: asyncio.Queue):
    """
    Queue에서 작업을 받아 GPU 학습을 시뮬레이션하는 Worker.
    """
    while True:
        # Queue에 작업이 들어올 때까지 비동기적으로 대기
        model_name, training_time = await queue.get()
        
        print(f"  👷 Worker [{name}]가 '{model_name}' 작업을 시작합니다. (예상 시간: {training_time:.2f}초)")
        
        # GPU 연산 대기 시간 모사 (GIL 해제)
        await asyncio.sleep(training_time)
        
        print(f"  🎉 Worker [{name}]가 '{model_name}' 작업을 완료했습니다!")
        
        # Queue에 작업이 완료되었음을 알림
        queue.task_done()

# Producer: 동적으로 학습 요청을 생성하는 역할
async def producer(queue: asyncio.Queue):
    """
    불규칙한 간격으로 새로운 학습 작업을 Queue에 추가합니다.
    """
    model_list = ["ResNet", "BERT", "GPT-3", "DALL-E 2", "Stable Diffusion"]
    for model_name in model_list:
        # 0.1초 ~ 1.5초 사이의 랜덤한 시간 간격으로 새 요청이 들어오는 상황 모사
        await asyncio.sleep(random.uniform(0.1, 1.5))
        
        training_time = random.uniform(2, 4) # 모델 학습 시간
        await queue.put((model_name, training_time))
        print(f"➡️  새로운 학습 요청: '{model_name}' (작업 Queue에 추가됨)")

# 메인 로직 실행
async def main():
    start_time = time.time()
    
    # 작업들을 담을 Queue 생성
    task_queue = asyncio.Queue()
    
    # 2개의 GPU Worker를 생성하여 Queue를 주시하도록 함
    # 실제로는 GPU 1개에서 동시성으로 처리되지만, 여러 요청을 처리하는 개념을 보여주기 위함
    workers = [
        asyncio.create_task(gpu_worker(f"Worker-{i}", task_queue))
        for i in range(2)
    ]
    
    # Producer를 실행하여 Queue에 작업을 동적으로 추가
    producer_task = asyncio.create_task(producer(task_queue))
    
    # Producer가 모든 작업을 Queue에 넣을 때까지 기다림
    await producer_task
    print("\n--- 모든 요청이 Queue에 추가되었습니다. 남은 작업 완료를 기다립니다... ---\n")
    
    # Queue에 있는 모든 작업이 처리될 때까지 기다림
    await task_queue.join()
    
    # 모든 작업이 완료되었으므로 Worker들을 중지시킴
    for worker in workers:
        worker.cancel()
        
    end_time = time.time()
    print(f"\n✨ 모든 동적 작업 완료! 총 소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    asyncio.run(main())
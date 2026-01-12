import asyncio
import time

def blocking_cake_baker():
    """3초가 걸리는 무거운 동기 작업 (제빵사)"""
    print("  👨‍🍳 제빵사: 케이크 굽기 시작...")
    time.sleep(3)
    print("  👨‍🍳 제빵사: 케이크 완성!")
    return "초코 케이크"

async def take_phone_call():
    """1초가 걸리는 가벼운 비동기 작업 (매니저)"""
    print("  🤵 매니저: 전화 받기 시작...")
    await asyncio.sleep(1)
    print("  🤵 매니저: 전화 끊음.")

async def main():
    print("카페 오픈!")
    start_time = time.time()
    
    loop = asyncio.get_running_loop()

    # 1. run_in_executor는 Future를 반환하므로, 그대로 변수에 할당합니다. (create_task로 감싸지 않음)
    cake_future = loop.run_in_executor(None, blocking_cake_baker)
    
    # 2. 코루틴은 create_task로 Task 객체를 만듭니다.
    phone_task = asyncio.create_task(take_phone_call())

    # 3. gather는 Future와 Task를 모두 동시에 기다릴 수 있습니다.
    await asyncio.gather(cake_future, phone_task)

    end_time = time.time()
    print(f"모든 작업 완료! 총 소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import time
import random

async def fetch_crypto_price(crypto_name: str):
    """
    하나의 서버에 접속해 데이터를 가져오는 작업을 시뮬레이션하는 코루틴.
    """
    print(f"🌐 '{crypto_name}' 서버에 가격 요청 시작...")
    
    # 네트워크 응답 대기 시간을 1~3초 사이의 랜덤 시간으로 시뮬레이션
    delay = random.uniform(1, 3)
    await asyncio.sleep(delay)
    
    # 서버로부터 받은 데이터라고 가정
    price = random.randint(1000, 50000)
    
    print(f"✅ '{crypto_name}' 가격 응답 받음! (소요 시간: {delay:.2f}초)")
    return {crypto_name: price}

async def main():
    """
    메인 로직: 여러 작업을 생성하고 동시에 실행
    """
    start_time = time.monotonic()
    
    # 1. 실행할 작업들을 정의 (아직 실행되지는 않음)
    cryptos = ["Bitcoin", "Ethereum", "Solana"]
    
    # 2. create_task로 각 작업을 '실행 예약'하고 리스트에 담음
    #    - 이 코드는 각 작업을 시작하라고 지시만 내리고, 바로 다음 코드로 넘어감 (기다리지 않음)
    tasks = [
        asyncio.create_task(fetch_crypto_price(crypto)) 
        for crypto in cryptos
    ]
    print("--- 모든 서버에 동시 요청 완료 ---")

    # 3. asyncio.gather로 리스트에 있는 모든 작업들이 끝날 때까지 기다림
    #    - 모든 셰프의 요리가 끝날 때까지 기다리는 것과 같음
    results = await asyncio.gather(*tasks)

    end_time = time.monotonic()
    
    print("\n--- 최종 결과 ---")
    print(results)
    print(f"⏱️ 총 소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    asyncio.run(main())
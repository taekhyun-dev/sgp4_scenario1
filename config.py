# 시뮬레이션 전반에 사용되는 설정값들을 정의하는 파일입니다.

# --- Satellite Local Training Settings ---
LOCAL_EPOCHS = 10              # 각 위성이 로컬 학습을 수행할 에포크 수
FEDPROX_MU = 0.01             # FedProx 하이퍼파라미터 (mu)

# --- Ground Station & IoT Settings ---
IOT_FLYOVER_THRESHOLD_DEG = 30.0  # 워커 위성이 이 고도각 이상으로 IoT 클러스터 상공을 통과할 때 학습을 시작합니다.
GS_FLYOVER_THRESHOLD_DEG = 10.0   # 마스터 위성이 이 고도각 이상으로 지상국 상공을 통과할 때 통신을 시작합니다.

# --- Federated Learning Aggregation Policy (Ground Station) ---
AGGREGATION_STALENESS_THRESHOLD = 5 # 글로벌 모델 버전과 취합 대상 클러스터 모델 버전 간의 최대 차이 허용치 (Staleness)

NUM_CLIENTS = 50       # 클라이언트 수
DIRICHLET_ALPHA = 0.5  # Non-IID 강도를 중간 정도로 설정
BATCH_SIZE = 128
NUM_WORKERS = 4        # 사용자의 환경에 맞게 조절
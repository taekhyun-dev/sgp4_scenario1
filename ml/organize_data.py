import os
import urllib.request
import shutil
from tqdm import tqdm

# ==========================================
# ⚠️ 본인의 val 폴더 경로가 맞는지 다시 한 번 확인해주세요!
VAL_DIR = "/home/octolab-rtx4090/.data/imagenet/ILSVRC/Data/CLS-LOC/val"
# ==========================================

def organize_validation_set(val_dir):
    print(f"📂 Validation Directory: {val_dir}")
    
    if not os.path.exists(val_dir):
        print("❌ Error: 해당 경로가 존재하지 않습니다.")
        return

    # 이미 폴더가 정리되어 있는지 확인 (폴더가 많으면 중단)
    first_level_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    if len(first_level_dirs) > 100:
        print("✅ 이미 클래스별로 정리된 것 같습니다. 작업을 중단합니다.")
        return

    # 1. 정답 라벨 파일(valprep.sh) 다운로드
    label_url = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
    print("⬇️  Label mapping script 다운로드 중...")
    
    try:
        with urllib.request.urlopen(label_url) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return

    # 2. 파싱 및 이동 작업
    print("📦 이미지 이동 시작...")
    
    lines = content.split('\n')
    
    # 현재 val 폴더에 있는 이미지 파일 리스트 확인 (확장자 대소문자 무시)
    current_files = set([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    print(f" - 현재 폴더 내 이미지 파일 수: {len(current_files)}")

    count = 0
    
    for line in tqdm(lines):
        parts = line.strip().split()
        
        # [수정] 'mv'로 시작하는 줄만 파싱 (mkdir 줄은 무시하고 os.makedirs로 처리)
        # 예: mv ILSVRC2012_val_00000001.JPEG n01440764/
        if len(parts) >= 2 and parts[0] == 'mv':
            file_name = parts[1]
            folder_name = parts[2].replace('/', '') # 뒤의 슬래시 제거
            
            # 소스 파일 경로
            src_path = os.path.join(val_dir, file_name)
            
            # 파일이 실제로 있을 때만 이동
            if os.path.exists(src_path):
                # 타겟 폴더 생성 (없으면 생성)
                target_dir = os.path.join(val_dir, folder_name)
                os.makedirs(target_dir, exist_ok=True)
                
                # 이동
                dst_path = os.path.join(target_dir, file_name)
                shutil.move(src_path, dst_path)
                count += 1
            
    print(f"\n✅ 완료! 총 {count}개의 이미지를 정리했습니다.")

if __name__ == "__main__":
    organize_validation_set(VAL_DIR)
import os
import modal
from datetime import datetime

# 1. Access Modal Volume & Folder v3
vol_name = "retinanet-v3-vol"
remote_dir = "retinanet_results_v3"
remote_file_name = "retinanet_formatted_results_v3.csv"

# 2. Setup Local Save Path & Timestamp
local_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(local_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_filename = f"retinanet_train_log_{timestamp}.csv"
save_filepath = os.path.join(local_dir, save_filename)

print("="*60)
print(f"📥 DOWNLOADING RETINANET (V3) TRAIN LOG FROM MODAL ({vol_name})")
print("="*60)

try:
    vol = modal.Volume.from_name(vol_name)
    remote_path = f"{remote_dir}/{remote_file_name}"
    
    print(f"🔄 Reading file from Modal: {remote_path}")
    file_bytes = b"".join(vol.read_file(remote_path))
    
    with open(save_filepath, "wb") as f:
        f.write(file_bytes)
        
    print(f"✅ Tải tệp log train RetinaNet V3 thành công!")
    print(f"📁 Lưu tại: {save_filepath}")
    print("="*60)
except Exception as e:
    print(f"❌ Lỗi khi tải log từ Modal (V3): {e}. Đang thử tìm với thư mục retinanet_results_v2...")
    try:
        remote_path_alt = f"retinanet_results_v2/{remote_file_name}"
        file_bytes = b"".join(vol.read_file(remote_path_alt))
        with open(save_filepath, "wb") as f:
            f.write(file_bytes)
        print(f"✅ Tải tệp log train RetinaNet thành công từ đường dẫn thay thế!")
        print(f"📁 Lưu tại: {save_filepath}")
    except Exception as ex:
        print(f"❌ Không thể tải log RetinaNet: {ex}")

import os
import modal

# 1. Khởi tạo Volume V4 hoàn toàn mới (Đảm bảo chạy sạch từ Epoch 1)
volume = modal.Volume.from_name("yolo11-v4-vol", create_if_missing=True)

# 2. Khai báo App Modal
app = modal.App(name="yolo11-training-v4")

# 3. Docker Image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics",
        "roboflow",
        "pandas"
    )
)

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="a100",     
    timeout=86400,  # 24 hours max timeout
)
def train():
    from roboflow import Roboflow
    from ultralytics import YOLO
    import shutil
    from pathlib import Path
    import os
    import time
    import json
    import pandas as pd

    # Thiết lập thư mục làm việc
    os.makedirs("/workspace", exist_ok=True)
    os.chdir("/workspace")

    # Đường dẫn lưu trữ V4
    storage_dir = Path("/storage/yolo11_results_v4")
    storage_weights_dir = storage_dir / "weights"
    storage_weights_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_times_file = storage_dir / "epoch_times.json"
    
    # Nạp thời gian epoch nếu có resume
    epoch_times = []
    if epoch_times_file.exists():
        try:
            with open(epoch_times_file, "r") as f:
                epoch_times = json.load(f)
            print(f"🔄 Loaded {len(epoch_times)} historical epoch times.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load epoch times: {e}")

    # 1. Tải Dataset Version 5 từ Roboflow
    print("📥 Downloading dataset version 5 from Roboflow (YOLOv8 format)...")
    rf = Roboflow(api_key="FndM2MNPaVHUVbzQmpMn")
    project = rf.workspace("nckhcict2025").project("completed-project")
    version = project.version(5)
    dataset = version.download("yolov8") 
    print(f"✅ Dataset v5 downloaded to: {dataset.location}")
    
    data_yaml_path = f"{dataset.location}/data.yaml"

    epoch_start_time = 0

    def sync_and_format_results(is_final=False):
        """Đồng bộ và format results.csv. Map thời gian theo số epoch tuyệt đối."""
        workspace_csv = Path("/workspace/runs/YOLO11n_640_v4/results.csv")
        if not workspace_csv.exists():
            return

        # Sao chép file CSV gốc của Ultralytics
        shutil.copy(workspace_csv, storage_dir / "results.csv")
        
        try:
            df = pd.read_csv(workspace_csv)
            df.columns = df.columns.str.strip()
            
            formatted_data = {
                "Epoch": df["epoch"],
                "Train loss": df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"],
                "Class loss": df["train/cls_loss"],
                "Box loss": df["train/box_loss"],
                "GIoU": "-",
                "Learning rate": df["lr/pg1"],
                "Val Precision": df["metrics/precision(B)"],
                "Val Recall": df["metrics/recall(B)"],
                "Val mAP50": df["metrics/mAP50(B)"],
                "Val mAP50-95": df["metrics/mAP50-95(B)"],
            }
            
            formatted_df = pd.DataFrame(formatted_data)
            
            # Map thời gian chính xác theo Epoch (1-indexed)
            times = []
            for ep in formatted_df["Epoch"]:
                ep_num = int(ep)
                if ep_num <= len(epoch_times) and epoch_times[ep_num - 1] is not None:
                    times.append(epoch_times[ep_num - 1])
                else:
                    times.append(None)
                    
            formatted_df["Time(s)"] = times
            formatted_df.to_csv(storage_dir / "yolo11_formatted_results_v4.csv", index=False)

            if is_final and len(formatted_df) > 0:
                print(f"📌 Total epochs in results.csv: {len(formatted_df)}")
                print(f"📌 Last epoch: {int(formatted_df['Epoch'].iloc[-1])}")
        except Exception as e:
            print(f"⚠️ Error formatting results CSV: {e}")

    def on_train_epoch_start(trainer):
        nonlocal epoch_start_time
        epoch_start_time = time.time()

    def on_fit_epoch_end(trainer):
        nonlocal epoch_start_time, epoch_times
        duration = time.time() - epoch_start_time
        
        current_epoch = trainer.epoch + 1
        while len(epoch_times) < current_epoch:
            epoch_times.append(None)
        epoch_times[current_epoch - 1] = round(duration, 2)
        
        # 1. Lưu thời gian từng epoch
        with open(epoch_times_file, "w") as f:
            json.dump(epoch_times, f)
            
        # 2. Sao lưu checkpoints
        workspace_weights = Path("/workspace/runs/YOLO11n_640_v4/weights")
        if workspace_weights.exists():
            if (workspace_weights / "last.pt").exists():
                shutil.copy(workspace_weights / "last.pt", storage_weights_dir / "last.pt")
            if (workspace_weights / "best.pt").exists():
                shutil.copy(workspace_weights / "best.pt", storage_weights_dir / "best.pt")
        
        # 3. Đồng bộ checkpoint log theo thời gian thực (Bảo vệ dữ liệu khi gặp sự cố)
        sync_and_format_results(is_final=False)
            
        volume.commit()
        print(f"💾 Epoch {current_epoch} completed in {duration:.1f}s. Saved & committed to Volume.")

    # 2. Khởi tạo mô hình
    last_checkpoint = storage_weights_dir / "last.pt"
    is_resuming = last_checkpoint.exists()

    if is_resuming:
        print(f"🔄 Found existing checkpoint at {last_checkpoint}. Resuming YOLO11n training...")
        model = YOLO(str(last_checkpoint))
    else:
        print("🚀 Initializing new YOLO11n model (Train from Epoch 1)...")
        model = YOLO('yolo11n.pt')

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # 3. Tiến hành huấn luyện
    print("🔥 Starting training YOLO11n_640_v4 (Max: 80, Patience: 10, Size: 640)...")
    try:
        if is_resuming:
            results = model.train(
                data=data_yaml_path,
                resume=True
            )
        else:
            results = model.train(
                data=data_yaml_path,
                epochs=80,
                patience=10,
                imgsz=640,
                batch=16,
                project="/workspace/runs",
                name="YOLO11n_640_v4"
            )
        print("🎉 YOLO11n training loop finished!")
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        sync_and_format_results(is_final=True)
        volume.commit()
        raise e

    # 4. Hậu xử lý: Copy toàn bộ plots, đồ thị và chốt file CSV cuối cùng
    runs_dir = Path("/workspace/runs/YOLO11n_640_v4")
    if runs_dir.exists():
        for ext in ["*.png", "*.jpg", "*.csv"]:
            for file_p in runs_dir.glob(ext):
                shutil.copy(file_p, storage_dir / file_p.name)
                
    # Format lần cuối cùng khi training loop đã đóng hoàn toàn
    sync_and_format_results(is_final=True)
                
    volume.commit()
    print(f"🎉 HOÀN TẤT! Toàn bộ kết quả và weights YOLO11n_640_v4 đã được lưu tại: {storage_dir}")

@app.local_entrypoint()
def main():
    print("🚀 Đang gửi job huấn luyện YOLO11n V4 lên Modal (GPU A100, Epochs 80, Patience 10)...")
    train.remote()

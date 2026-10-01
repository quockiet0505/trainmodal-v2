import os
import modal

# 1. Define Persistent Storage Volume for V2
volume = modal.Volume.from_name("yolo11-v2-vol", create_if_missing=True)

# 2. Define App
app = modal.App(name="yolo11-training-v2")

# 3. Define Docker Image
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

    # Ensure working directory is writeable
    os.makedirs("/workspace", exist_ok=True)
    os.chdir("/workspace")

    # Paths setup for V2
    storage_dir = Path("/storage/yolo11_results_v2")
    storage_weights_dir = storage_dir / "weights"
    storage_weights_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_times_file = storage_dir / "epoch_times.json"
    
    # Load persistent epoch times if resuming
    epoch_times = []
    if epoch_times_file.exists():
        try:
            with open(epoch_times_file, "r") as f:
                epoch_times = json.load(f)
            print(f"🔄 Loaded {len(epoch_times)} historical epoch times.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load epoch times: {e}")

    # 1. Download Dataset Version 5 from Roboflow
    print("📥 Downloading dataset version 5 from Roboflow (YOLOv8 format)...")
    rf = Roboflow(api_key="FndM2MNPaVHUVbzQmpMn")
    project = rf.workspace("nckhcict2025").project("completed-project")
    version = project.version(5)
    dataset = version.download("yolov8") 
    print(f"✅ Dataset v5 downloaded to: {dataset.location}")
    
    data_yaml_path = f"{dataset.location}/data.yaml"

    # Define custom callbacks for epoch timing and real-time checkpoint backup
    epoch_start_time = 0

    def on_train_epoch_start(trainer):
        nonlocal epoch_start_time
        epoch_start_time = time.time()

    def on_train_epoch_end(trainer):
        nonlocal epoch_start_time, epoch_times
        duration = time.time() - epoch_start_time
        epoch_times.append(round(duration, 2))
        
        # Save epoch times to volume
        with open(epoch_times_file, "w") as f:
            json.dump(epoch_times, f)
            
        # Copy checkpoints from workspace to storage volume
        workspace_weights = Path("/workspace/runs/YOLO11n_640_v2/weights")
        if workspace_weights.exists():
            if (workspace_weights / "last.pt").exists():
                shutil.copy(workspace_weights / "last.pt", storage_weights_dir / "last.pt")
            if (workspace_weights / "best.pt").exists():
                shutil.copy(workspace_weights / "best.pt", storage_weights_dir / "best.pt")
        
        # Copy current results.csv to storage and format in real-time
        workspace_csv = Path("/workspace/runs/YOLO11n_640_v2/results.csv")
        if workspace_csv.exists():
            shutil.copy(workspace_csv, storage_dir / "results.csv")
            try:
                df = pd.read_csv(workspace_csv)
                df.columns = df.columns.str.strip()
                
                formatted_data = {
                    "Epoch": df["epoch"],
                    "Train loss": df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"],
                    "Class loss": df["train/cls_loss"],
                    "Box loss": df["train/box_loss"],
                    "Learning rate": df["lr/pg1"],
                    "Val Precision": df["metrics/precision(B)"],
                    "Val Recall": df["metrics/recall(B)"],
                    "Val mAP50": df["metrics/mAP50(B)"],
                    "Val mAP50-95": df["metrics/mAP50-95(B)"],
                }
                
                formatted_df = pd.DataFrame(formatted_data)
                min_len = min(len(formatted_df), len(epoch_times))
                formatted_df = formatted_df.iloc[:min_len].copy()
                formatted_df["Time(s)"] = epoch_times[:min_len]
                
                formatted_df.to_csv(storage_dir / "yolo11_formatted_results_v2.csv", index=False)
            except Exception as e:
                print(f"⚠️ Error formatting results CSV during epoch: {e}")
            
        volume.commit()
        print(f"💾 Epoch {len(epoch_times)} completed in {duration:.1f}s. Saved & committed to Volume.")

    # 2. Check if resuming from checkpoint
    last_checkpoint = storage_weights_dir / "last.pt"
    is_resuming = last_checkpoint.exists()

    if is_resuming:
        print(f"🔄 Found existing checkpoint at {last_checkpoint}. Resuming YOLO11n training...")
        model = YOLO(str(last_checkpoint))
    else:
        print("🚀 Initializing new YOLO11n model...")
        model = YOLO('yolo11n.pt')

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 3. Train the model (Max Epochs: 80, Patience: 10, Imgsz: 640, GPU: A100)
    print("🔥 Starting training YOLO11n_640_v2...")
    try:
        if is_resuming:
            results = model.train(
                data=data_yaml_path,
                resume=True
            )
        else:
            results = model.train(
                data=data_yaml_path,
                epochs=80,       # Max epochs = 80
                patience=10,     # Patience = 10
                imgsz=640,       # Image size = 640x640
                batch=16,        
                project="/workspace/runs",
                name="YOLO11n_640_v2"
            )
        print("🎉 YOLO11n training completed!")
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        volume.commit()
        raise e

    # 4. Final post-processing and copying plots
    runs_dir = Path("/workspace/runs/YOLO11n_640_v2")
    if runs_dir.exists():
        for ext in ["*.png", "*.jpg"]:
            for img_file in runs_dir.glob(ext):
                shutil.copy(img_file, storage_dir / img_file.name)
                
    volume.commit()
    print(f"🎉 HOÀN TẤT! toàn bộ log và weights YOLO11n_640_v2 đã lưu trên Modal Volume tại: {storage_dir}")

@app.local_entrypoint()
def main():
    print("🚀 Đang gửi job huấn luyện YOLO11n V2 lên Modal (GPU A100, Epochs 80, Patience 10)...")
    train.remote()

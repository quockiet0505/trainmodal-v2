import os
import modal

# 1. Define Persistent Storage Volume for V3
volume = modal.Volume.from_name("retinanet-v3-vol", create_if_missing=True)

# 2. Define App
app = modal.App(name="retinanet-training-v3")

# 3. Define Docker Image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "numpy<2.0.0",
        "torch",
        "torchvision",
        "torchmetrics",
        "roboflow",
        "pycocotools",
        "pandas",
        "tqdm"
    )
)

@app.function(
    image=image,
    volumes={"/storage": volume},
    gpu="a100",     
    timeout=86400,  # 24 hours max timeout
)
def train():
    import torch
    import torchvision
    import torchvision.transforms.functional as F
    from torch.utils.data import DataLoader
    from roboflow import Roboflow
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import os
    import shutil
    import time
    from pathlib import Path
    
    os.makedirs("/workspace", exist_ok=True)
    os.chdir("/workspace")
    
    print("📥 Downloading dataset version 5 (COCO format) from Roboflow...")
    rf = Roboflow(api_key="FndM2MNPaVHUVbzQmpMn")
    project = rf.workspace("nckhcict2025").project("completed-project")
    version = project.version(5)
    
    dataset = version.download("coco") 
    data_dir = dataset.location
    print(f"✅ Dataset v5 downloaded to: {data_dir}")

    # Custom Coco Detection Dataset mapping bbox format (x,y,w,h) -> (x1,y1,x2,y2) scaled to 640x640
    class CustomCocoDetection(torchvision.datasets.CocoDetection):
        def __init__(self, root, annFile):
            super().__init__(root, annFile)
            
        def __getitem__(self, index):
            img, targets = super().__getitem__(index)
            # Ensure 3-channel RGB
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            
            # Explicit 640x640 resize
            img = img.resize((640, 640))
            img = F.to_tensor(img)
            
            scale_x = 640.0 / orig_w
            scale_y = 640.0 / orig_h
            
            boxes = []
            labels = []
            for t in targets:
                x, y, w, h = t['bbox']
                if w > 0 and h > 0:
                    x1 = max(0.0, min(640.0, float(x * scale_x)))
                    y1 = max(0.0, min(640.0, float(y * scale_y)))
                    x2 = max(0.0, min(640.0, float((x + w) * scale_x)))
                    y2 = max(0.0, min(640.0, float((y + h) * scale_y)))
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(t['category_id'])
                
            if len(boxes) > 0:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
                
            target = {"boxes": boxes, "labels": labels}
            return img, target

    def collate_fn(batch):
        return tuple(zip(*batch))

    # DataLoaders
    train_dataset = CustomCocoDetection(
        root=f"{data_dir}/train", 
        annFile=f"{data_dir}/train/_annotations.coco.json"
    )
    val_dataset = CustomCocoDetection(
        root=f"{data_dir}/valid", 
        annFile=f"{data_dir}/valid/_annotations.coco.json"
    )

    # Verify category label mapping directly from dataset
    coco_train = train_dataset.coco
    categories = sorted(coco_train.dataset['categories'], key=lambda x: x['id'])
    print(f"\n📋 Kiểm tra danh sách {len(categories)} lớp:")
    for cat in categories[:3]:
        print(f"   - ID {cat['id']}: {cat['name']}")
    print("   ...")
    for cat in categories[-3:]:
        print(f"   - ID {cat['id']}: {cat['name']}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 Training RetinaNet_ResNet50_640_v2 on GPU device: {device}")

    # 32 categories + 1 background = 33
    NUM_CLASSES = 33 
    
    print("🚀 Initializing model retinanet_resnet50_fpn...")
    try:
        from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            num_classes=NUM_CLASSES,
            weights_backbone=RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            min_size=640,
            max_size=640
        )
    except Exception:
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            num_classes=NUM_CLASSES,
            min_size=640,
            max_size=640
        )

    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    EPOCHS = 80      # Max Epochs = 80
    PATIENCE = 10    # Patience = 10

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Explicitly define IoU thresholds 0.50 -> 0.95, step 0.05
    iou_steps = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05).tolist()]
    val_metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=iou_steps,
        class_metrics=True,
        extended_summary=True
    )
    
    storage_dir = Path("/storage/retinanet_results_v3")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    last_checkpoint_path = storage_dir / "last.pth"
    best_checkpoint_path = storage_dir / "best.pth"
    
    start_epoch = 1
    best_map50_95 = 0.0
    patience_counter = 0
    logs = []
    
    # Resume from checkpoint if present
    if last_checkpoint_path.exists():
        print(f"🔄 Found checkpoint at {last_checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_map50_95 = checkpoint['best_map50_95']
            patience_counter = checkpoint['patience_counter']
            logs = checkpoint.get('logs', [])
            print(f"🔄 Resumed successfully from epoch {start_epoch} (Best mAP50-95: {best_map50_95:.4f})")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Starting fresh...")

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # 1. TRAIN LOOP
        model.train()
        train_loss_total = 0.0
        train_loss_cls = 0.0
        train_loss_box = 0.0
        
        for images, targets in tqdm(train_loader, desc="Training"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss_total += losses.item()
            train_loss_cls += loss_dict['classification'].item()
            train_loss_box += loss_dict['bbox_regression'].item()
            
        num_batches = len(train_loader)
        avg_loss = train_loss_total / num_batches
        avg_cls = train_loss_cls / num_batches
        avg_box = train_loss_box / num_batches

        # 2. VALIDATION LOOP
        model.eval()
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = list(image.to(device) for image in images)
                preds = model(images)
                
                preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                val_metric.update(preds_cpu, targets_cpu)
                
        metrics_result = val_metric.compute()
        map50 = metrics_result['map_50'].item()
        map50_95 = metrics_result['map'].item()
        
        prec_tensor = metrics_result.get("precision")
        if prec_tensor is not None and prec_tensor.numel() > 0:
            valid_p = prec_tensor[prec_tensor >= 0]
            precision = valid_p.mean().item() if valid_p.numel() > 0 else 0.0
        else:
            precision = 0.0

        rec_tensor = metrics_result.get("recall")
        if rec_tensor is not None and rec_tensor.numel() > 0:
            valid_r = rec_tensor[rec_tensor >= 0]
            recall = valid_r.mean().item() if valid_r.numel() > 0 else 0.0
        else:
            recall = 0.0

        val_metric.reset()  # Reset cache for next epoch
        
        print(f"Train Loss: {avg_loss:.4f} (Cls: {avg_cls:.4f}, Box: {avg_box:.4f})")
        print(f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val mAP50: {map50:.4f} | Val mAP50-95: {map50_95:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # Update Learning Rate Scheduler
        epoch_time = time.time() - epoch_start_time
        
        log_entry = {
            "Epoch": epoch,
            "Train loss": round(avg_loss, 5),
            "Class loss": round(avg_cls, 5),
            "Box loss": round(avg_box, 5),
            "Learning rate": current_lr,
            "Val Precision": round(precision, 5),
            "Val Recall": round(recall, 5),
            "Val mAP50": round(map50, 5),
            "Val mAP50-95": round(map50_95, 5),
            "Time(s)": round(epoch_time, 2)
        }
        
        logs.append(log_entry)
        
        df = pd.DataFrame(logs)
        df.to_csv(storage_dir / "retinanet_formatted_results_v3.csv", index=False)
        
        # Early Stopping check & checkpoint save
        is_best = map50_95 > best_map50_95
        if is_best:
            best_map50_95 = map50_95
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_map50_95': best_map50_95,
                'patience_counter': patience_counter,
                'logs': logs
            }, best_checkpoint_path)
            print(f"🌟 Saved new best.pth (mAP50-95: {best_map50_95:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ Patience: {patience_counter}/{PATIENCE}")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map50_95': best_map50_95,
            'patience_counter': patience_counter,
            'logs': logs
        }, last_checkpoint_path)
        
        volume.commit()
        print(f"💾 Checkpoints and logs committed to volume for Epoch {epoch}.")
            
        if patience_counter >= PATIENCE:
            print("🛑 Early Stopping triggered! Stopping training.")
            break
            
    print(f"🎉 RetinaNet_ResNet50_640_v2 training completed! Results saved to: {storage_dir}")

@app.local_entrypoint()
def main():
    print("🚀 Sending RetinaNet_ResNet50_640_v2 training job to Modal (GPU A100)...")
    train.remote()

# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION (COCO BENCHMARK)

- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn (32 lớp)
- **Thời gian**: 04/10/2026 01:17:48
- **Chuẩn đánh giá**: pycocotools COCOeval, 1481 ảnh, protocol `729c28d00fa5c8e2a5b3e56c1dd1fa0efd45bbf05d4f7e0d67cc0307a44068a1`
- **P/R/F1**: suy ra từ COCOeval matches tại conf=0.55, IoU=0.5; các cột mAP/AR là COCO metrics.

| Mô hình | Precision | Recall | F1 | mAP@50 | mAP@50-95 | mAP@75 | AR@100 | Patience | GFLOPs | Parameters | Latency | FPS | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RetinaNet | 0.9094 | 0.9568 | 0.9325 | 0.9624 | 0.7961 | 0.8793 | 0.8449 | 10 | 86.99 | 32.86 | 21.85 | 45.76 | Hoàn tất đánh giá |

*Ghi chú: Không đồng nhất Precision/Recall tại một ngưỡng cố định với COCO AP/AR tích phân.*

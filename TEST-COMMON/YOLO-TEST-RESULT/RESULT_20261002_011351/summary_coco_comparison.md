# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION (COCO BENCHMARK)

- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn (32 lớp)
- **Thời gian**: 01/10/2026 18:15:57
- **Chuẩn đánh giá**: pycocotools COCOeval, 1481 ảnh, protocol `c5d2b02af047a7896464b115f59b28752fc1526ca6fadcb899ca943bb73cd4a8`
- **P/R/F1**: suy ra từ COCOeval matches tại conf=0.55, IoU=0.5; các cột mAP/AR là COCO metrics.

| Mô hình | Precision | Recall | F1 | mAP@50 | mAP@50-95 | mAP@75 | AR@100 | Patience | GFLOPs | Parameters | Latency | FPS | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv11 | 0.9710 | 0.9604 | 0.9657 | 0.9818 | 0.8601 | 0.9418 | 0.8958 | 10 | 3.24 | 2.60 | 23.03 | 43.42 | Hoàn tất đánh giá |

*Ghi chú: Không đồng nhất Precision/Recall tại một ngưỡng cố định với COCO AP/AR tích phân.*

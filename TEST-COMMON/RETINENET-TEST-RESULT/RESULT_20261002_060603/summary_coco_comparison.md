# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION (COCO BENCHMARK)

- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn (32 lớp)
- **Thời gian**: 01/10/2026 23:08:36
- **Chuẩn đánh giá**: pycocotools COCOeval, 1481 ảnh, protocol `c5d2b02af047a7896464b115f59b28752fc1526ca6fadcb899ca943bb73cd4a8`
- **P/R/F1**: suy ra từ COCOeval matches tại conf=0.55, IoU=0.5; các cột mAP/AR là COCO metrics.

| Mô hình | Precision | Recall | F1 | mAP@50 | mAP@50-95 | mAP@75 | AR@100 | Patience | GFLOPs | Parameters | Latency | FPS | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RetinaNet | 0.9023 | 0.9611 | 0.9308 | 0.9646 | 0.7990 | 0.8847 | 0.8489 | 10 | 86.99 | 32.86 | 32.97 | 30.33 | Hoàn tất đánh giá |

*Ghi chú: Không đồng nhất Precision/Recall tại một ngưỡng cố định với COCO AP/AR tích phân.*

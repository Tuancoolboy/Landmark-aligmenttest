# TinyFace alignment benchmark trên Google Colab

Notebook điều khiển: `notebooks/tinyface_alignment_colab.ipynb`.

Notebook clone repository vào `/content/Landmark-aligmenttest`, tải ZIP TinyFace từ Google Drive, chuẩn hóa dữ liệu vào `data/tinyface`, rồi benchmark bốn pipeline `dfa_mobilenet`, `dfa_resnet50`, `scrfd10g` và `mediapipe`.

## Chuẩn bị

1. Mở notebook bằng Google Colab và chọn GPU runtime.
2. Revoke HF token đã từng gửi trong chat; tạo token mới chỉ có quyền đọc.
3. Trong Colab mở Secrets (biểu tượng chìa khóa), tạo secret tên `HF_TOKEN`.
4. Cho phép notebook truy cập Google Drive.

Dataset không được commit vào GitHub. Kết quả/checkpoint được lưu tại `MyDrive/tinyface_alignment_results` để có thể resume.

## Đánh giá

Notebook chạy single-view, không flip TTA. Báo cáo gồm Rank-1/5/20 theo protocol TinyFace, coverage, latency p50/p95 và lỗi theo size bin. Vì TinyFace không có ground-truth landmark, không báo cáo NME/PCK.

Các file sinh ra trong Drive:

- `results.json`
- `summary.csv`
- `benchmark_report.md`
- `checkpoints/<pipeline>.npz`


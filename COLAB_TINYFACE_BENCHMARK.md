# TinyFace alignment benchmark — Jupyter điều khiển Modal

Notebook điều khiển: `notebooks/tinyface_alignment_colab.ipynb`. Notebook chạy trong Jupyter/local và gọi Modal GPU; không dùng `google.colab`.

Notebook clone repository vào thư mục hiện tại, tải ZIP TinyFace từ Google Drive, upload vào Modal Volume `/data/tinyface`, rồi benchmark bốn pipeline `dfa_mobilenet`, `dfa_resnet50`, `scrfd10g` và `mediapipe`.

## Chuẩn bị

1. Cài và đăng nhập Modal CLI (`modal token new`).
2. Revoke HF token đã từng gửi trong chat; tạo token mới chỉ có quyền đọc.
3. Mở notebook bằng Jupyter/VS Code notebook tại workspace đã clone.
4. Google Drive ZIP cần cho phép tải public; notebook dùng `gdown`.

Dataset không được commit vào GitHub. Dataset nằm trong Modal Volume `cvlface-tinyface-data`; model/cache và kết quả nằm trong các Modal Volume do `modal_app.py` quản lý.

## Đánh giá

Notebook gọi `modal_app.py` chạy single-view, không flip TTA. Báo cáo gồm Rank-1/5/20, coverage, latency p50/p95 và lỗi theo size bin. Vì TinyFace không có ground-truth landmark, không báo cáo NME/PCK.

Kết quả trả về local:

- `results.json`
- `summary.csv`
- `benchmark_report.md`
- `benchmarks/tinyface_alignment/last_modal_result.json`

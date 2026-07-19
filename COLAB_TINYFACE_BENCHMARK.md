# TinyFace alignment benchmark — Local Jupyter

Notebook chạy trực tiếp trong kernel Jupyter có GPU; không dùng `google.colab` hoặc Modal.

Notebook tải ZIP TinyFace vào repository local rồi benchmark bốn pipeline `dfa_mobilenet`, `dfa_resnet50`, `scrfd10g` và `mediapipe`.

## Chuẩn bị

1. Mở notebook bằng Jupyter/VS Code tại workspace đã clone.
2. Chọn kernel có GPU CUDA.
3. Google Drive ZIP cần cho phép tải public; notebook dùng `gdown`.
4. Nếu cần HF token, đặt qua biến môi trường `HF_TOKEN`; không hard-code.

Dataset, model cache, checkpoint và kết quả đều nằm local và đã được loại khỏi Git.

## Đánh giá

Notebook chạy single-view, không flip TTA. Báo cáo gồm Rank-1/5/20, coverage và latency p50/p95. Vì TinyFace không có ground-truth landmark, không báo cáo NME/PCK.

Kết quả trong thư mục `results/`:

- `results.json`
- `summary.csv`
- `benchmark_report.md`
- `<pipeline>.npz` checkpoint để resume

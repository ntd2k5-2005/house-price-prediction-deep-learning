# Dự đoán giá nhà bằng Deep Learning

Project xây dựng hệ thống dự đoán giá nhà tại một thành phố bằng Python, PyTorch và Streamlit, sử dụng bộ dữ liệu Kaggle House Prices - Advanced Regression Techniques.

## 1. Mô tả bài toán

Mục tiêu là dự đoán `SalePrice` dựa trên các đặc trưng như diện tích, số phòng, vị trí, năm xây dựng và các thuộc tính liên quan. Project thực hiện tiền xử lý dữ liệu, tạo đặc trưng mới, huấn luyện ba mô hình deep learning và so sánh bằng MSE, MAE, RMSE, R2, thời gian huấn luyện và thời gian suy luận.

## 2. Yêu cầu đề bài được đáp ứng

- Chuẩn hóa dữ liệu numeric và one-hot encoding dữ liệu categorical.
- Tạo đặc trưng mới, bao gồm tương tác giữa diện tích và vị trí.
- Thiết kế và huấn luyện ANN baseline.
- Xây dựng mô hình tiên tiến FT-Transformer Lite cho dữ liệu bảng.
- Đề xuất ANN chuẩn hóa đặc trưng thích nghi.
- Đánh giá bằng metrics trên giá gốc sau inverse log-transform.
- Xuất bảng, biểu đồ, prediction và file tóm tắt thực nghiệm cho Word/PPT.
- Có app demo bằng Streamlit.

## 3. Cấu trúc thư mục

```text
data/
  raw/
    train.csv
    test.csv
    sample_submission.csv
    data_description.txt
  processed/
outputs/
  models/
  figures/
  results/
  reports/
src/
  models/
app/
notebooks/
main.py
requirements.txt
README.md
```

## 4. Chuẩn bị dữ liệu

Tải bộ dữ liệu Kaggle House Prices - Advanced Regression Techniques và đặt file chính vào:

```text
data/raw/train.csv
```

File `train.csv` là bắt buộc vì có cột `SalePrice`. File `test.csv` không dùng để tính MSE/MAE vì không có `SalePrice`; chỉ phù hợp cho demo/submission nếu mở rộng.

## 5. Tạo môi trường ảo trên Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tất cả package được cài trong `.venv`, không cần GPU/CUDA.

## 6. Chạy toàn bộ pipeline

```powershell
.\.venv\Scripts\python.exe main.py --mode all
```

Chạy nhanh để kiểm tra pipeline:

```powershell
.\.venv\Scripts\python.exe main.py --mode all --epochs 1
```

Các mode hỗ trợ:

```powershell
.\.venv\Scripts\python.exe main.py --mode train
.\.venv\Scripts\python.exe main.py --mode evaluate
.\.venv\Scripts\python.exe main.py --mode all
```

## 7. Chạy Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

App có bốn tab: dự đoán giá nhà, so sánh mô hình, biểu đồ thực nghiệm và thông tin mô hình.

## 8. Ba mô hình

- **ANN baseline**: MLP truyền thống gồm Linear, ReLU, BatchNorm và Dropout.
- **FT-Transformer Lite**: mỗi feature sau preprocessing được xem như một token scalar, sau đó đưa qua TransformerEncoder gọn để học quan hệ giữa các đặc trưng.
- **ANN chuẩn hóa đặc trưng thích nghi**: mô hình đề xuất gồm AdaptiveFeatureNormalization, FeatureGate và MLP hồi quy.

## 9. Metrics

Các metrics được tính trên giá gốc sau khi inverse log-transform bằng `np.expm1`:

- MSE: Mean Squared Error.
- MAE: Mean Absolute Error.
- RMSE: căn bậc hai của MSE.
- R2: hệ số xác định.
- training_time_seconds: thời gian huấn luyện.
- inference_time_seconds: thời gian suy luận trên tập test.

## 10. Output phục vụ báo cáo

```text
outputs/models/
  ann_baseline.pt
  ft_transformer_lite.pt
  adaptive_ann.pt
  best_model.pt
  preprocessor.joblib
outputs/results/
  model_comparison.csv
  predictions_best_model.csv
outputs/figures/
  loss_ann_baseline.png
  loss_ft_transformer_lite.png
  loss_adaptive_ann.png
  metrics_comparison.png
  actual_vs_predicted_best_model.png
  residual_plot_best_model.png
outputs/reports/
  experiment_summary.md
```

## 11. Lưu ý

- Không tạo dữ liệu giả và không tạo kết quả giả.
- Không dùng Kaggle `test.csv` để đánh giá chính vì không có `SalePrice`.
- Nếu thiếu `data/raw/train.csv`, project sẽ không train/evaluate và sẽ yêu cầu đặt file đúng vị trí.
- Không nên kết luận mô hình cải tiến tốt hơn nếu bảng metrics không chứng minh điều đó.

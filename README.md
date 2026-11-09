# Credit Card Fraud Detection

Dự án phát hiện gian lận thẻ tín dụng bằng XGBoost. API dự đoán giao dịch, giải thích các feature ảnh hưởng bằng SHAP, và có giao diện web demo.

## Công nghệ

- Python, Pandas, NumPy
- Scikit-learn, XGBoost, SHAP, Joblib
- FastAPI, Pydantic, Uvicorn
- HTML, CSS, JavaScript

## Cài đặt

Chạy trong PowerShell từ thư mục gốc dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy API và giao diện

```powershell
Set-Location .\fraud_api
python -m uvicorn main:app --reload
```

- Giao diện: <http://127.0.0.1:8000/ui/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- API health check: <http://127.0.0.1:8000/fraud/>

## Endpoints

| Method | Endpoint | Chức năng |
|---|---|---|
| GET | `/` | Kiểm tra API và liên kết giao diện |
| GET | `/fraud/` | Health check |
| POST | `/fraud/predict` | Trả nhãn và xác suất dự đoán |
| POST | `/fraud/explain` | Trả dự đoán và top 3 SHAP impacts |
| GET | `/fraud/model-info` | Xem phiên bản model và metrics |

Request cho hai endpoint dự đoán gồm `Time`, `V1`–`V28`, và `Amount`. Gửi `Time` và `Amount` ở dạng raw; model pipeline tự scale chúng. `V1`–`V28` là feature ẩn danh của dataset.

Trong giao diện, chọn **Load Normal sample** hoặc **Load Fraud sample** để điền một giao dịch mẫu từ dataset. Có thể mở nhóm **Model features** để xem hoặc sửa `V1`–`V28`.

## Retrain model

Từ thư mục gốc, chạy:

```powershell
python -m training.train_model
```

Máy có GPU CUDA tương thích có thể dùng:

```powershell
python -m training.train_model --device cuda
```

Có thể chỉ định dataset khác:

```powershell
python -m training.train_model --data "path\to\transactions.csv" --device cpu
```

Mỗi lần chạy lưu model, threshold, metadata và kết quả đánh giá vào thư mục phiên bản riêng trong `artifacts/fraud_detection_runs/`. Retrain tạo candidate artifacts; để API sử dụng model mới, cần promote các artifacts đó vào `fraud_api/models`.

## Cấu trúc dự án

- `data/`: dataset raw và dữ liệu xử lý cho phân tích
- `notebook/`: EDA, thử nghiệm model và explainability
- `training/`: script retrain có thể chạy lại
- `artifacts/`: output của từng lần retrain
- `fraud_api/`: FastAPI, schemas, routers, services và model đang phục vụ
- `frontend/`: giao diện HTML/CSS/JavaScript

## Lưu ý

Đây là dự án học tập dùng dataset thẻ tín dụng đã ẩn danh. Các feature trong dataset không đại diện cho dữ liệu giao dịch ngân hàng đầy đủ trong môi trường thực tế.

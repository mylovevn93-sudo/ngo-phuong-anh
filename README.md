# Hệ Thống Phát Hiện Giao Dịch Bất Thường (Bank Anomaly Detection)

Ứng dụng web được xây dựng bằng **Streamlit** nhằm phát hiện các giao dịch ngân hàng bất thường dựa trên thuật toán học máy không giám sát **Isolation Forest** (Rừng Cô Lập) từ thư viện `scikit-learn`.

Dự án này được tối ưu hóa từ Notebook phân tích dữ liệu gốc `phathien_batthuong.ipynb` sang một giao diện Dashboard tương tác hiện đại và chuyên nghiệp.

## ✨ Tính năng chính

- 📊 **Phân tích tổng quan dữ liệu (EDA):** Trực quan hóa số lượng giao dịch theo giờ và biểu đồ phân phối số tiền giao dịch bằng Plotly tương tác.
- ⚙️ **Cấu hình mô hình động:** Cho phép người dùng tinh chỉnh trực tiếp các tham số của thuật toán `Isolation Forest` (như tỉ lệ nhiễm bẩn `contamination`, số lượng cây quyết định `n_estimators`, v.v.) ngay trên thanh Sidebar.
- 🔍 **Huấn luyện & Hiển thị kết quả:**
  - Nhận diện giao dịch bất thường (`is_anomaly == True`).
  - Lọc ra 25% các trường hợp bất thường có điểm rủi ro cao nhất làm danh sách **Khẩn cấp** cần xử lý ngay.
  - Trực quan hóa ranh giới quyết định của mô hình bằng biểu đồ Scatter 2D sinh động.
- 📥 **Xuất báo cáo:** Hỗ trợ tải xuống danh sách giao dịch bất thường (CSV) và tệp tin khẩn cấp dưới dạng Excel (`khan_cap.xlsx`).
- 🔬 **Tra cứu chi tiết (Transaction Inspector):** Hỗ trợ tìm kiếm giao dịch theo ID hoặc mã khách hàng để lý giải nguyên nhân tại sao giao dịch bị cảnh báo.

---

## 🚀 Hướng dẫn chạy ứng dụng Local

### 1. Yêu cầu hệ thống
Đảm bảo máy tính của bạn đã cài đặt **Python** (phiên bản từ 3.8 đến 3.11).

### 2. Cài đặt các thư viện cần thiết
Mở Terminal/Command Prompt trong thư mục chứa dự án và chạy lệnh:
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy Web App
Chạy lệnh Streamlit để khởi động ứng dụng trên trình duyệt web:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại địa chỉ local: `http://localhost:8501`.

---

## ☁️ Hướng dẫn Deploy lên Streamlit Cloud

Để ứng dụng có thể chạy trực tuyến và chia sẻ với người khác, hãy làm theo các bước sau:

1. **Đưa mã nguồn lên GitHub:**
   - Tạo một repository mới trên tài khoản GitHub của bạn.
   - Commit các tệp sau lên repository đó:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `transactions_Q1_demo.csv` (Để hệ thống tự động tải file demo mặc định)

2. **Triển khai ứng dụng trên Streamlit Cloud:**
   - Truy cập trang web [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
   - Nhấn nút **"Create app"** (hoặc **"New app"**).
   - Chọn đúng repository chứa dự án, nhánh (`main` hoặc `master`), và chỉ định tệp chính chạy ứng dụng là `app.py`.
   - Bấm nút **"Deploy!"** và đợi hệ thống cài đặt các thư viện từ `requirements.txt` trong khoảng 1-2 phút.

Ứng dụng của bạn sẽ được kích hoạt trực tuyến dưới dạng tên miền có dạng `https://<tên-ứng-dụng>.streamlit.app/`!

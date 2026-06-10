import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import io
import os

# Cấu hình trang Streamlit với giao diện rộng rãi và tiêu đề chuyên nghiệp
st.set_page_config(
    page_title="Hệ thống Phát hiện Giao dịch Bất thường trong Kiểm toán nội bộ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy biến để tạo giao diện cao cấp (Glassmorphism & Neon Accent)
st.markdown("""
<style>
    /* Gradient header và kiểu chữ */
    .main-title {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Thiết kế thẻ Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.2);
    }
    
    /* Custom style cho các tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- PHẦN TIÊU ĐỀ CHÍNH -----------------
st.markdown('<h1 class="main-title">🛡️ HỆ THỐNG PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ứng dụng trí tuệ nhân tạo (Isolation Forest) phát hiện các hành vi gian lận và giao dịch bất thường trong hệ thống ngân hàng.</p>', unsafe_allow_html=True)

# ----------------- SIDEBAR CẤU HÌNH MÔ HÌNH -----------------
st.sidebar.header("⚙️ Cấu Hình Mô Hình & Dữ Liệu")

# Chọn nguồn dữ liệu
data_source = st.sidebar.radio(
    "Nguồn dữ liệu đầu vào:",
    ("Sử dụng dữ liệu Demo (Q1)", "Tải lên tệp dữ liệu tùy chỉnh (.csv)")
)

df = None
default_file = "transactions_Q1_demo.csv"

if data_source == "Sử dụng dữ liệu Demo (Q1)":
    if os.path.exists(default_file):
        df = pd.read_csv(default_file, parse_dates=["transaction_date"], dayfirst=False)
        st.sidebar.success(f"Loaded demo file: `{default_file}` successfully.")
    else:
        st.sidebar.error(f"Không tìm thấy file mặc định `{default_file}` trong thư mục ứng dụng. Vui lòng tải file lên.")
else:
    uploaded_file = st.sidebar.file_uploader("Tải lên file giao dịch CSV của bạn:", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["transaction_date"], dayfirst=False)
            st.sidebar.success("Tải tệp thành công!")
        except Exception as e:
            st.sidebar.error(f"Lỗi đọc file: {e}")

# Sidebar cấu hình Isolation Forest
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Tham số Isolation Forest")

contamination = st.sidebar.slider(
    "Tỷ lệ bất thường giả định (Contamination):",
    min_value=0.001,
    max_value=0.10,
    value=0.01,
    step=0.001,
    format="%.3f"
)

n_estimators = st.sidebar.number_input(
    "Số lượng cây (n_estimators):",
    min_value=50,
    max_value=1000,
    value=200,
    step=50
)

random_state = st.sidebar.number_input(
    "Random State (Đảm bảo kết quả nhất quán):",
    min_value=0,
    max_value=1000,
    value=42
)

# ----------------- XỬ LÝ CHÍNH -----------------
if df is not None:
    # 1. Tiền xử lý dữ liệu EDA
    # Tạo biến gio_giao_dich và co_nhan_vien
    df['gio_giao_dich'] = df['transaction_date'].dt.hour
    df['co_nhan_vien'] = df['is_employee'].astype(int)

    # Layout Dashboard với 3 Tabs
    tab_eda, tab_model, tab_inspect = st.tabs(["📊 Phân Tích Tổng Quan (EDA)", "🔍 Mô Hình & Kết Quả Bất Thường", "🔬 Tra Cứu Giao Dịch"])

    # ---------------- TAB 1: EDA ----------------
    with tab_eda:
        st.subheader("📈 Tổng Quan Đặc Điểm Tập Dữ Liệu")
        
        # Thống kê nhanh dưới dạng cards
        total_txns = len(df)
        total_amount = df['amount'].sum()
        avg_amount = df['amount'].mean()
        employee_txns = df['is_employee'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng Số Giao Dịch", f"{total_txns:,}")
        with col2:
            st.metric("Tổng Tiền Giao Dịch", f"{total_amount:,} VND")
        with col3:
            st.metric("Giao Dịch Trung Bình", f"{avg_amount:,.2f} VND")
        with col4:
            st.metric("Giao Dịch bởi Nhân Viên", f"{employee_txns:,}")
            
        st.markdown("---")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Phân bố giao dịch theo giờ trong ngày")
            txn_hours = df['gio_giao_dich'].value_counts().sort_index().reset_index()
            txn_hours.columns = ['Giờ', 'Số lượng giao dịch']
            
            fig_hour = px.bar(
                txn_hours, 
                x='Giờ', 
                y='Số lượng giao dịch',
                labels={'Giờ': 'Giờ trong ngày', 'Số lượng giao dịch': 'Số lượng giao dịch'},
                color='Số lượng giao dịch',
                color_continuous_scale='dense',
                template='plotly_dark'
            )
            fig_hour.update_layout(height=400)
            st.plotly_chart(fig_hour, use_container_width=True)
            
        with col_right:
            st.markdown("#### Thống kê mô tả số tiền giao dịch")
            st.dataframe(df['amount'].describe().reset_index().rename(columns={'index': 'Chỉ số mô tả', 'amount': 'Giá trị (VND)'}), use_container_width=True)
            
            # Biểu đồ boxplot số tiền giao dịch để xem ngoại lai sơ bộ
            fig_box = px.box(
                df, 
                y='amount', 
                points="outliers",
                title="Biểu đồ Boxplot Phân Phối Số Tiền (Thang logarit)",
                log_y=True,
                template='plotly_dark',
                color_discrete_sequence=['#6a11cb']
            )
            fig_box.update_layout(height=280)
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Preview Dữ Liệu
        st.markdown("#### Xem trước dữ liệu thô (20 dòng đầu)")
        st.dataframe(df.head(20), use_container_width=True)

    # ---------------- TAB 2: MODEL & RESULTS ----------------
    with tab_model:
        st.subheader("🤖 Kết Quả Huấn Luyện Mô Hình Học Máy")
        
        # Chuẩn bị dữ liệu huấn luyện
        X = df[["amount", "gio_giao_dich", "co_nhan_vien"]]
        
        with st.spinner("Đang chuẩn hóa dữ liệu và huấn luyện mô hình Isolation Forest..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Khởi tạo và fit mô hình
            iso = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                max_samples="auto",
                random_state=random_state,
                n_jobs=-1
            )
            iso.fit(X_scaled)
            
            # Thêm kết quả vào DataFrame gốc
            df["anomaly_score"] = iso.decision_function(X_scaled)
            df["is_anomaly"] = iso.predict(X_scaled) == -1
            
        # Tính toán kết quả bất thường
        df_bat_thuong = df[df['is_anomaly'] == True]
        total_anoms = len(df_bat_thuong)
        
        # Các trường hợp khẩn cấp (thuộc 25% nhóm có điểm rủi ro cao nhất của danh sách bất thường)
        score_25th_quantile = df_bat_thuong['anomaly_score'].quantile(0.25)
        df_khan_cap = df_bat_thuong[df_bat_thuong['anomaly_score'] < score_25th_quantile]
        total_urgent = len(df_khan_cap)
        
        # Hiển thị metrics huấn luyện
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Tổng Số Bất Thường Phát Hiện", f"{total_anoms:,}", delta=f"{total_anoms/total_txns*100:.2f}% tỷ lệ thực tế", delta_color="inverse")
        with col_m2:
            st.metric("Tỷ Lệ Contamination Thiết Lập", f"{contamination*100:.2f}%")
        with col_m3:
            st.metric("Trường Hợp Khẩn Cấp (Top 25% Rủi Ro)", f"{total_urgent:,}", "Cần xử lý gấp", delta_color="inverse")
        with col_m4:
            st.metric("Điểm Phân Ngưỡng Khẩn Cấp", f"{score_25th_quantile:.6f}")
            
        st.markdown("---")
        
        # Tạo Biểu đồ trực quan hóa bất thường bằng Scatter 2D
        st.markdown("#### Trực quan hóa ranh giới phát hiện bất thường")
        st.write("Biểu đồ thể hiện mối quan hệ giữa **Giờ giao dịch** và **Số tiền**, được tô màu theo trạng thái phân loại bất thường.")
        
        # Vẽ biểu đồ Plotly Scatter
        fig_scatter = px.scatter(
            df.sample(n=min(10000, len(df)), random_state=42), # Sample để biểu đồ mượt mà nếu dữ liệu quá lớn
            x="gio_giao_dich",
            y="amount",
            color="is_anomaly",
            color_discrete_map={False: "#4361ee", True: "#ff0054"},
            labels={"gio_giao_dich": "Giờ Giao Dịch", "amount": "Số Tiền Giao Dịch (VND)", "is_anomaly": "Bất Thường?"},
            hover_data=["transaction_id", "location", "is_employee", "anomaly_score"],
            log_y=True,
            title="Biểu đồ phân tách giao dịch (Màu Hồng Đỏ là Bất thường)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Tạo hai tabs hiển thị bảng kết quả
        tab_list_anom, tab_list_urgent = st.tabs(["🔴 Tất cả giao dịch bất thường", "🚨 Trường hợp khẩn cấp cần xử lý ngay"])
        
        with tab_list_anom:
            st.markdown(f"Danh sách **{total_anoms}** giao dịch được mô hình đánh dấu là bất thường:")
            st.dataframe(df_bat_thuong.sort_values(by="anomaly_score"), use_container_width=True)
            
            # Xuất file CSV/Excel cho toàn bộ danh sách bất thường
            csv_anom = df_bat_thuong.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải Xuất toàn bộ giao dịch bất thường (CSV)",
                data=csv_anom,
                file_name="all_anomalies.csv",
                mime="text/csv"
            )
            
        with tab_list_urgent:
            st.markdown(f"Danh sách **{total_urgent}** giao dịch khẩn cấp nhất (Score < {score_25th_quantile:.6f}):")
            st.dataframe(df_khan_cap.sort_values(by="anomaly_score"), use_container_width=True)
            
            # Tạo Excel buffer để xuất file Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_khan_cap.to_excel(writer, index=False, sheet_name="Khan Cap")
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Tải xuống danh sách Khẩn Cấp (Excel - khan_cap.xlsx)",
                data=excel_data,
                file_name="khan_cap.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ---------------- TAB 3: TRANSACTION INSPECTOR ----------------
    with tab_inspect:
        st.subheader("🔬 Tra Cứu Chi Tiết Từng Giao Dịch")
        st.write("Nhập mã giao dịch (Transaction ID) hoặc mã khách hàng (Customer Hash) để kiểm tra điểm rủi ro cụ thể và lý giải nguyên nhân.")
        
        search_option = st.selectbox("Tìm kiếm theo:", ["Mã giao dịch (transaction_id)", "Mã khách hàng (customer_id_hash)"])
        search_query = st.text_input("Nhập từ khóa tìm kiếm (Ví dụ: `TXN00047828` hoặc `TXN_ANOM_MIX_0483`):")
        
        if search_query:
            if search_option == "Mã giao dịch (transaction_id)":
                result = df[df['transaction_id'].str.contains(search_query.strip(), case=False, na=False)]
            else:
                result = df[df['customer_id_hash'].str.contains(search_query.strip(), case=False, na=False)]
                
            if len(result) > 0:
                st.success(f"Tìm thấy {len(result)} kết quả tương thích!")
                for idx, row in result.iterrows():
                    with st.expander(f"Giao dịch: {row['transaction_id']} - Khách hàng: {row['customer_id_hash'][:8]}... - Số tiền: {row['amount']:,} VND"):
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.write(f"**Mã giao dịch:** {row['transaction_id']}")
                            st.write(f"**Ngày giao dịch:** {row['transaction_date']}")
                            st.write(f"**Loại giao dịch:** {row['transaction_type']}")
                            st.write(f"**Kênh thực hiện:** {row['channel']}")
                            st.write(f"**Ngân hàng đối tác:** {row['counterparty_bank']}")
                        with col_r:
                            st.write(f"**Chi nhánh thực hiện:** {row['location']}")
                            st.write(f"**Nhân viên thực hiện?** {'Có' if row['is_employee'] else 'Không'}")
                            st.write(f"**Giờ thực hiện:** {row['gio_giao_dich']}h")
                            
                            # Hiển thị điểm bất thường và trạng thái
                            score = row['anomaly_score']
                            is_anom = row['is_anomaly']
                            
                            if is_anom:
                                is_urgent = score < score_25th_quantile
                                status_text = "⚠️ KHẨN CẤP" if is_urgent else "🔴 BẤT THƯỜNG"
                                st.error(f"**Trạng thái phân loại:** {status_text} (Score: {score:.6f})")
                            else:
                                st.success(f"**Trạng thái phân loại:** ✅ BÌNH THƯỜNG (Score: {score:.6f})")
                                
                        # Lý giải lý do bất thường
                        if is_anom:
                            st.markdown("#### 📑 Đánh giá rủi ro sơ bộ:")
                            reasons = []
                            if row['amount'] > df['amount'].quantile(0.99):
                                reasons.append(f"- Giao dịch có giá trị cực lớn ({row['amount']:,} VND), thuộc nhóm 1% lớn nhất hệ thống.")
                            if row['gio_giao_dich'] < 6 or row['gio_giao_dich'] > 22:
                                reasons.append(f"- Giao dịch được thực hiện vào khung giờ đêm muộn/sáng sớm ({row['gio_giao_dich']}h), ngoài giờ hoạt động thông thường.")
                            if row['is_employee']:
                                reasons.append("- Giao dịch được thực hiện bởi nhân viên ngân hàng (đặc trưng tăng thêm độ nhạy cảm kiểm soát).")
                                
                            if len(reasons) > 0:
                                for r in reasons:
                                    st.write(r)
                            else:
                                st.write("- Giao dịch bị mô hình đánh giá là bất thường dựa trên sự kết hợp đồng thời của các yếu tố (số tiền lớn kết hợp khung giờ và đối tượng giao dịch).")
            else:
                st.warning("Không tìm thấy giao dịch nào khớp với từ khóa tìm kiếm của bạn.")
else:
    st.info("👋 Chào mừng bạn! Hãy tải tệp dữ liệu giao dịch định dạng CSV lên hoặc nhấn chọn nút dữ liệu Demo ở cột bên trái để bắt đầu phân tích.")

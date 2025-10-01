import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import time
from datetime import datetime
import numpy as np
import random
import concurrent.futures
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from dask_ml.preprocessing import Categorizer
# Cấu hình trang
st.set_page_config(
    page_title="Tiki Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .st-bw {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #FF6B6B;
        text-align: center;
        padding: 20px;
    }
    .stPlotlyChart {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo session state
if 'start_idx' not in st.session_state:
    st.session_state.start_idx = 0
    st.session_state.batch_size = random.uniform(20,40)
    st.session_state.total_records = None
    st.session_state.current_data = pd.DataFrame()

def connect_db():
    conn = psycopg2.connect(
        database="data_tiki",
        user="nhanpham",
        password="123456",
        host="localhost",
        port="5432"
    )
    return conn

def get_total_records():
    if st.session_state.total_records is None:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM data_product")
        st.session_state.total_records = cursor.fetchone()[0]
        conn.close()
    return st.session_state.total_records

def get_data_batch():
    conn = connect_db()
    query = f"""
    SELECT productid, name, brandname, originalprice, price, 
           discount, discountrate, quantitysold, ratingaverage
    FROM data_product
    LIMIT {st.session_state.batch_size} 
    OFFSET {st.session_state.start_idx}
    """
    df_batch = pd.read_sql(query, conn)
    conn.close()

    # Làm sạch dữ liệu: loại bỏ giá trị âm hoặc NaN
    df_batch = df_batch.dropna(subset=["price", "ratingaverage", "quantitysold"])
    df_batch = df_batch[(df_batch["price"] >= 0) & 
                        (df_batch["ratingaverage"] > 0) & 
                        (df_batch["quantitysold"] >= 0)]
    return df_batch


def run_apriori_analysis(df_sample):
    """
    Hàm thực hiện phân tích giỏ hàng bằng thuật toán Apriori.
    """
    df_basket = df_sample.groupby(['brandname', 'productid'])['quantitysold'].sum().unstack().fillna(0)
    df_basket = df_basket > 0  # Chuyển thành boolean

    if not df_basket.empty:
        # Tính tập mục phổ biến
        frequent_itemsets = apriori(df_basket, min_support=0.05, use_colnames=True)
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(map(str, list(x))))

        if not frequent_itemsets.empty:
            # Tính luật kết hợp
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=10, support_only=True)
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        else:
            rules = None
        return frequent_itemsets, rules
    else:
        return None, None

# Header
st.title("🛍️ Tiki E-commerce Analytics Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Bộ lọc")
    price_range = st.slider(
        "Khoảng giá (VNĐ)",
        min_value=float(0),
        max_value=float(10000000),
        value=(0.0, 10000000.0)
    )
    min_rating = st.slider("Đánh giá tối thiểu", 0.0, 5.0, 0.0)
    
    st.markdown("---")
    st.markdown("### Thông tin cập nhật")
    last_update = st.empty()

# Load và cập nhật dữ liệu
new_batch = get_data_batch()
st.session_state.current_data = pd.concat([st.session_state.current_data, new_batch], ignore_index=True)
df = st.session_state.current_data

# Loại bỏ sản phẩm có `ratingaverage == 0`
# df = df[df["ratingaverage"] > 0]

# Lọc dữ liệu
df_filtered = df[
    (df['price'] >= price_range[0]) & 
    (df['price'] <= price_range[1]) &
    (df['ratingaverage'] >= min_rating)
]

# Metrics tổng quan
st.markdown("### 📊 Chỉ số tổng quan")
metric1, metric2, metric3, metric4 = st.columns(4)
with metric1:
    st.metric("📦 Tổng số sản phẩm", f"{len(df_filtered):,}")
with metric2:
    st.metric("🛒 Tổng số lượng bán", f"{df_filtered['quantitysold'].sum():,}")
with metric3:
    st.metric("💰 Giá trung bình", f"{df_filtered['price'].mean():,.0f} VNĐ")
with metric4:
    st.metric("⭐ Đánh giá trung bình", f"{df_filtered['ratingaverage'].mean():.2f}/5")

st.markdown("---")

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📊 Phân Tích Giá và Thương Hiệu", "⭐ Đánh Giá và Tỷ Lệ Giảm Giá", "📈 Chiến Lược Marketing"])

# Tab 1: Phân tích giá và thương hiệu
with tab1:
    st.markdown("### 📊 Phân Tích Giá và Thương Hiệu")
    col1, col2 = st.columns(2)

    with col1:
        # Biểu đồ phân bố giá
        fig1 = px.histogram(
            df_filtered, 
            x='price',
            title='📊 Phân bố giá sản phẩm',
            labels={'price': 'Giá (VNĐ)', 'count': 'Số lượng'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig1.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, l=40, r=40, b=40)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Biểu đồ Top thương hiệu bán chạy nhất
        top_brands = df_filtered.groupby('brandname')['quantitysold'].sum().sort_values(ascending=True).tail(10)
        fig2 = go.Figure(go.Bar(
            x=top_brands.values,
            y=top_brands.index,
            orientation='h',
            marker_color='#4ECDC4'
        ))
        fig2.update_layout(
            title='🏢 Top 10 thương hiệu bán chạy nhất',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, l=200, r=40, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Phân tích đánh giá và tỷ lệ giảm giá
with tab2:
    st.markdown("### ⭐ Đánh Giá và Tỷ Lệ Giảm Giá")
    col1, col2 = st.columns(2)

    with col1:
        # Biểu đồ mối quan hệ giữa rating và số lượng bán
        fig3 = px.scatter(
            df_filtered, 
            x='ratingaverage', 
            y='quantitysold',
            title='⭐ Mối quan hệ giữa đánh giá và số lượng bán',
            labels={'ratingaverage': 'Điểm đánh giá', 'quantitysold': 'Số lượng bán'},
            color='price',
            size='quantitysold',
            color_continuous_scale='Viridis'
        )
        fig3.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, l=40, r=40, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Biểu đồ ảnh hưởng của tỷ lệ giảm giá đến số lượng bán
        fig4 = px.scatter(
            df_filtered, 
            x='discountrate', 
            y='quantitysold',
            title='💯 Ảnh hưởng của tỷ lệ giảm giá đến số lượng bán',
            labels={'discountrate': 'Tỷ lệ giảm giá (%)', 'quantitysold': 'Số lượng bán'},
            color='price',
            size='quantitysold',
            color_continuous_scale='Viridis'
        )
        fig4.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, l=40, r=40, b=40)
        )
        st.plotly_chart(fig4, use_container_width=True)

# Tab 3: Chiến lược marketing
with tab3:
    st.markdown("### 📈 Thuật Toán Hỗ Trợ Chiến Lược Marketing")
    st.markdown("Sử dụng các thuật toán để phân tích dữ liệu và đưa ra chiến lược kinh doanh.")

    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # 1️⃣ Nút chọn mẫu dữ liệu
    if st.button("🗂 Chọn mẫu dữ liệu"):
        with st.spinner("Đang lấy mẫu dữ liệu..."):
            max_samples = 2000  # Giới hạn số mẫu
            df_sample = df_filtered.sample(n=min(len(df_filtered), max_samples), random_state=42)
            st.session_state["df_sample"] = df_sample  # Lưu vào session để dùng lại
        st.success(f"Đã chọn {len(df_sample)} mẫu dữ liệu!")

    # 2️⃣ Phân nhóm K-Means
    if st.button("🔍 Phân Nhóm K-Means"):
        st.session_state.is_processing = True
    try:
        if "df_sample" in st.session_state:
            with st.spinner("Đang phân nhóm sản phẩm..."):
                df_sample = st.session_state["df_sample"]
                features = ["price", "ratingaverage", "quantitysold"]
                df_clean = df_sample.dropna(subset=features)
                X = df_clean[features]

                if len(X) >= 3:
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    df_clean["cluster"] = kmeans.fit_predict(X)

                    # Hiển thị biểu đồ
                    fig_kmeans = px.scatter_3d(
                        df_clean, x="price", y="ratingaverage", z="quantitysold",
                        color="cluster",
                        title="Phân Nhóm Sản Phẩm (K-Means Clustering)",
                        labels={"price": "Giá (VNĐ)", "ratingaverage": "Điểm đánh giá", "quantitysold": "Số lượng bán"}
                    )
                    st.plotly_chart(fig_kmeans, use_container_width=True)
                    st.success("Phân nhóm sản phẩm thành công!")
                else:
                    st.warning("Dữ liệu không đủ để phân nhóm. Hãy chọn lại dữ liệu với ít nhất 3 mẫu.")
    finally:
        st.session_state.is_processing = False

     # 3️⃣ Phân Tích Giỏ Hàng (Apriori)
    if st.button("🛒 Phân Tích Giỏ Hàng (Apriori)"):
        st.session_state.is_processing = True
        try:
            if "df_sample" in st.session_state:
                df_sample = st.session_state["df_sample"]

                with st.spinner("Đang phân tích giỏ hàng..."):
                    # Tạo executor để chạy phân tích giỏ hàng trong luồng riêng
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_apriori_analysis, df_sample)
                        frequent_itemsets, rules = future.result()

                    if frequent_itemsets is not None:
                        st.write("Frequent Itemsets:")
                        st.dataframe(frequent_itemsets)
                        st.markdown(""" **Giải nghĩa:** - **itemsets:** Các nhóm sản phẩm thường được mua cùng nhau. - **support:** Tần suất xuất hiện của nhóm sản phẩm trong tập dữ liệu (phần trăm giao dịch chứa nhóm sản phẩm). """)

                        if rules is not None:
                            st.write("### Các luật kết hợp phổ biến:")
                            st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
                            st.markdown(""" **Giải nghĩa:** - **antecedents:** Tập sản phẩm xuất hiện trước. - **consequents:** Tập sản phẩm được mua thêm khi đã mua tập antecedents. - **support:** Tỷ lệ giao dịch chứa cả antecedents và consequents. - **confidence:** Xác suất mua consequents khi đã mua antecedents. - **lift:** Mức độ liên kết giữa antecedents và consequents, giá trị >1 cho thấy có mối quan hệ mạnh. """)
                            st.success("Phân tích giỏ hàng thành công!")
                        else:
                            st.warning("Không tìm thấy tập mục phổ biến nào. Hãy thử giảm giá trị min_support.")
                    else:
                        st.warning("Dữ liệu giỏ hàng rỗng. Vui lòng kiểm tra bộ lọc hoặc nguồn dữ liệu!")
            else:
                st.warning("Bạn cần chọn mẫu dữ liệu trước khi chạy thuật toán Apriori!")
        finally:
            st.session_state.is_processing = False




# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ by Hữu Hậu</p>
    </div>
""", unsafe_allow_html=True)

# Cập nhật thời gian
last_update.text(f"Cập nhật lần cuối: {datetime.now().strftime('%H:%M:%S')}")

# Tăng index cho lần load tiếp theo
if st.session_state.start_idx < get_total_records():
    st.session_state.start_idx += st.session_state.batch_size

# Auto refresh
time.sleep(random.uniform(1,5))
st.rerun()
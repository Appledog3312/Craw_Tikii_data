import streamlit as st
import pandas as pd
import psycopg2
import time
import matplotlib.pyplot as plt
import random

# Kết nối tới PostgreSQL
def connect_db():
    return psycopg2.connect(
        dbname="data_tiki", user="nhanpham", password="123456", host="localhost", port="5432"
    )

# Lấy dữ liệu từ PostgreSQL
def fetch_data():
    conn = connect_db()
    query = "SELECT * FROM data_product"
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Hàm cập nhật ngẫu nhiên từ 10-50 dòng dữ liệu từ CSV vào bảng PostgreSQL
def update_random_data(file_path, num_rows=10):
    conn = connect_db()
    cursor = conn.cursor()

    df = pd.read_csv(file_path)
    sampled_df = df.sample(n=num_rows, replace=True)

    insert_query = """
    INSERT INTO data_product (
        productId, sellerId, name, brandName, 
        originalPrice, price, discount, discountRate, 
        quantitySold, ratingAverage
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (productId) DO NOTHING;
    """
    try:
        for _, row in sampled_df.iterrows():
            cursor.execute(insert_query, (
                row['productId'], row['sellerId'], row['name'], row['brandName'],
                row['originalPrice'], row['price'], row['discount'], row['discountRate'],
                row['quantitySold'], row['ratingAverage']
            ))
        conn.commit()
    except Exception as e:
        st.error(f"Lỗi khi cập nhật dữ liệu: {e}")
    finally:
        cursor.close()
        conn.close()


# Vẽ biểu đồ phân phối giá
def plot_price_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df['price'].dropna(), bins=20, color="orange", edgecolor="black")
    plt.title("Phân Phối Giá Sản Phẩm")
    plt.xlabel("Giá")
    plt.ylabel("Tần suất")
    st.pyplot(plt)

# Vẽ biểu đồ phân phối điểm đánh giá
def plot_rating_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df['ratingAverage'].dropna(), bins=10, color="green", edgecolor="black")
    plt.title("Phân Phối Điểm Đánh Giá")
    plt.xlabel("Điểm đánh giá")
    plt.ylabel("Tần suất")
    st.pyplot(plt)

# Main app
def main():
    st.title("Phân Tích Xu Hướng Khách Hàng")
    file_path = "dim_product.csv"  # Đường dẫn file CSV

    # Slider chọn số lượng dòng cập nhật mỗi 5 giây
    num_rows = st.slider("Số dòng cập nhật ngẫu nhiên mỗi 5 giây", 10, 50, 20)

    # Placeholder để cập nhật liên tục
    placeholder = st.empty()

    while True:
        # Cập nhật dữ liệu ngẫu nhiên
        update_random_data(file_path, num_rows)
        
        # Lấy dữ liệu từ PostgreSQL
        df = fetch_data()

        if not df.empty:
            with placeholder.container():
                st.subheader("Dữ Liệu Mới Nhất")
                st.dataframe(df.tail(10))  # Hiển thị 10 dòng mới nhất

                st.subheader("Phân Phối Giá Sản Phẩm")
                plot_price_distribution(df)

                st.subheader("Phân Phối Điểm Đánh Giá")
                plot_rating_distribution(df)
        else:
            st.warning("Không có dữ liệu trong bảng.")
        
        # Đợi 5 giây trước khi cập nhật tiếp
        time.sleep(5)

if __name__ == "__main__":
    main()


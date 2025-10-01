import os
import pandas as pd
import psycopg2

# Kết nối tới PostgreSQL
def connect_db():
    conn = psycopg2.connect(
        dbname="data_tiki", user="nhanpham", password="123456", host="localhost", port="5432"
    )
    return conn

# Làm sạch dữ liệu: giới hạn giá trị quá lớn
def clean_numeric_values(df):
    max_value = 999999999.99  # Giá trị tối đa cho NUMERIC(15, 2)
    for col in ['originalPrice', 'price', 'discount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Chuyển về dạng số
            df[col] = df[col].apply(lambda x: min(x, max_value) if pd.notna(x) else x)
    return df

# Tạo bảng PostgreSQL
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS data_product (
        productId BIGINT PRIMARY KEY,
        sellerId BIGINT,
        name TEXT,                -- Đổi VARCHAR thành TEXT
        brandName TEXT,          -- Đổi VARCHAR thành TEXT
        originalPrice NUMERIC(15, 2),
        price NUMERIC(15, 2),
        discount NUMERIC(15, 2),
        discountRate NUMERIC(5, 2),
        quantitySold INTEGER,
        ratingAverage NUMERIC(3, 2)
    );
    """
    try:
        cursor.execute(create_table_query)
        conn.commit()
        print("Bảng 'data_product' đã được tạo thành công.")
    except Exception as e:
        print("Lỗi khi tạo bảng:", e)
    finally:
        cursor.close()
        conn.close()

# Làm sạch dữ liệu: cắt chuỗi dài không hợp lệ
def clean_string_values(df):
    string_columns = ['name', 'brandName']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')  # Thay NaN bằng chuỗi rỗng
            df[col] = df[col].astype(str)  # Chuyển tất cả về dạng chuỗi
    return df

# Chèn dữ liệu vào PostgreSQL
def insert_data_from_csv(file_path):
    conn = connect_db()
    cursor = conn.cursor()

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)
    df = clean_numeric_values(df)  # Làm sạch dữ liệu số
    df = clean_string_values(df)   # Làm sạch dữ liệu chuỗi

    insert_query = """
    INSERT INTO data_product (
        productId, sellerId, name, brandName, 
        originalPrice, price, discount, discountRate, 
        quantitySold, ratingAverage
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (productId) DO NOTHING;
    """
    try:
        for _, row in df.iterrows():
            cursor.execute(insert_query, (
                row['productId'], row['sellerId'], row['name'], row['brandName'],
                row['originalPrice'], row['price'], row['discount'], row['discountRate'],
                row['quantitySold'], row['ratingAverage']
            ))
        conn.commit()
        print("Dữ liệu đã được chèn thành công vào bảng 'data_product'.")
    except Exception as e:
        print("Lỗi khi chèn dữ liệu:", e)
    finally:
        cursor.close()
        conn.close()

# Main function
def main():
    file_path = "dim_product.csv"  # Đường dẫn file CSV
    create_table()
    insert_data_from_csv(file_path)

if __name__ == "__main__":
    main()
import psycopg2
from psycopg2 import OperationalError

def create_connection():
    try:
        # Kết nối tới PostgreSQL
        conn = psycopg2.connect(
            user="nhanpham",        # Thay bằng tên user của bạn
            password="123456", # Thay bằng mật khẩu đúng của user
            host="localhost",        # Địa chỉ máy chủ (localhost cho máy tính cục bộ)
            port="5432",             # Cổng mặc định của PostgreSQL
            database="tikifinal"     # Tên cơ sở dữ liệu
        )
        print("Kết nối thành công!")
        return conn
    except OperationalError as e:
        print(f"Lỗi kết nối: {e}")
        return None

# Kiểm tra kết nối
if __name__ == "__main__":
    conn = create_connection()
    if conn:
        # Nếu kết nối thành công, đóng kết nối sau khi kiểm tra
        conn.close()

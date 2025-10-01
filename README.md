# Craw_Tikii_data

## 📌 Giới thiệu
Đề tài: **Xây dựng website trực quan hóa dữ liệu từ sàn thương mại điện tử Tiki**  
Mục tiêu:
- Thu thập dữ liệu sản phẩm từ Tiki.
- Lưu trữ dữ liệu vào **PostgreSQL**.
- Xây dựng website trực quan hóa dữ liệu bằng **Flask API** và thư viện trực quan hóa (ECharts / Plotly).

## 🛠️ Công nghệ sử dụng
- **Python 3.x**
- **Flask** (REST API backend)
- **PostgreSQL** (cơ sở dữ liệu)
- **SQLAlchemy / psycopg2** (kết nối DB)
- **Pandas, Numpy** (xử lý dữ liệu)
- **ECharts / Plotly / Chart.js** (trực quan hóa)
- **BeautifulSoup / Requests** (crawl dữ liệu Tiki)

## 📂 Cấu trúc thư mục
```bash
├── src/                  # Mã nguồn chính
│   ├── crawl/            # Code crawler Tiki
│   ├── api/              # Flask API
│   ├── database/         # Kết nối PostgreSQL
│   ├── static/           # CSS, JS
│   └── templates/        # HTML (Flask render)
├── requirements.txt      # Thư viện Python cần cài
└── README.md             # Tài liệu dự án



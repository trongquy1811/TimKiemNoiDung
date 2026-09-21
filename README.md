# 📄 DocAnalyzer AI Pro - Tìm Kiếm Nội Dung & Phân Tích Tài Liệu Thông Minh

<p align="center">
  <b>Hệ thống phân tích tài liệu và hình ảnh đa phương thức sử dụng Streamlit, Gemini RAG & Vision API.</b>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.30+-red?style=for-the-badge&logo=streamlit" alt="Streamlit"></a>
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Gemini-AI%20Pro-orange?style=for-the-badge&logo=google-gemini" alt="Gemini"></a>
</p>

---

## ✨ Tính Năng Nổi Bật (Key Features)

### 🧠 Pipeline RAG Nâng Cao & Tìm Kiếm Ngữ Nghĩa (Hybrid RAG)
* **Semantic Search (Gemini Embeddings):** Sử dụng mô hình `models/text-embedding-004` của Google để trích xuất Vector Embeddings mật độ cao, giúp khớp ngữ cảnh theo ý nghĩa thay vì chỉ khớp từ khóa.
* **Xếp hạng Cosine Similarity:** Xếp hạng các đoạn văn bản (chunks) chính xác và nhanh chóng dựa trên thuật toán Cosine Similarity tối ưu bằng NumPy.
* **Cơ chế Auto-Fallback:** Tự động chuyển sang thuật toán tìm kiếm theo từ khóa (Keyword Search) nếu mất kết nối hoặc hết hạn ngạch (quota) API Gemini, đảm bảo ứng dụng hoạt động không gián đoạn.

### 🖼️ Xử Lý PDF Quét & Đa Phương Thức (Smart Scanned PDF OCR)
* **Nhận diện trang PDF quét:** Tự động quét tài liệu PDF từng trang. Nếu phát hiện trang tài liệu dạng quét/ảnh (không trích xuất được text), hệ thống tự động tách xuất các tệp ảnh ẩn trong trang.
* **Trích xuất ảnh không phụ thuộc thư viện C++:** Sử dụng `pypdf.page.images` thuần Python để trích xuất ảnh trên các trang PDF scan và truyền trực tiếp vào **Gemini Vision API** mà không cần cài đặt Poppler/PDF2Image phức tạp.
* **Phân tích hình ảnh trực quan (Gemini Vision):** Hỗ trợ phân tích ảnh chụp, biểu đồ, hình ảnh y tế (như điện tâm đồ ECG), hình ảnh tài liệu...

### 🎨 Giao Diện Glassmorphism Sang Trọng & Tối Ưu State
* **Typography Outfit:** Tích hợp phông chữ hiện đại Google Fonts Outfit.
* **Bong bóng chat Glassmorphism:** Hiệu ứng làm mờ kính cường lực cao cấp, gradient chuyển màu xanh dương ấn tượng cho người dùng và màu slate tối cho AI.
* **Quản lý Streamlit State ổn định:** Khắc phục hoàn toàn lỗi rerun của Streamlit khi chat. Khu vực xem trước hình ảnh (Preview) và danh sách context được giữ nguyên trên màn hình.
* **Bảng điều khiển Sidebar linh hoạt:** Dễ dàng thay đổi API Key, chọn dòng mô hình Gemini (`gemini-2.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-exp`), tùy chỉnh Temperature, Tokens limit, số lượng Top-K context, dọn dẹp lịch sử chat và nhật ký câu hỏi.

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

* **Giao diện & Web App:** [Streamlit](https://streamlit.io/)
* **Mô hình Trí tuệ Nhân tạo (LLM & Vision):** [Google Generative AI SDK](https://ai.google.dev/) (`gemini-2.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-exp`, `models/text-embedding-004`)
* **Xử lý tài liệu & văn bản:** PyPDF (`pypdf`), Regex, Splitter
* **Xử lý số liệu & Hình ảnh:** NumPy, Pillow (`PIL`)
* **Ngôn ngữ lập trình:** Python 3.11+

---

## 📂 Cấu Trúc Dự Án (Project Structure)

```bash
TimKiemNoiDung/
├── app.py                   # Giao diện chính Streamlit Web App & Chat Interface
├── src/                     # Thư viện mô-đun hóa hỗ trợ backend
│   ├── __init__.py          # Export các hàm và hằng số chính của gói src
│   ├── config.py            # Đường dẫn tĩnh, danh sách model và cấu hình hệ thống
│   ├── dataset.py           # Bộ đọc và phân đoạn dữ liệu (PDF, Markdown, TXT)
│   ├── inference.py         # Hàm gọi API Gemini (Text QA & Gemini Vision OCR)
│   ├── preprocessing.py    # Thuật toán cắt nhỏ văn bản (Text Chunking with Overlap)
│   ├── rag.py               # Thuật toán RAG (Semantic Search & Keyword Search)
│   └── utils.py             # Hàm băm MD5 tệp và lưu trữ upload
├── uploads/                 # Thư mục chứa tài liệu upload và ảnh trích xuất tạm thời
├── requirements.txt         # Danh sách thư viện Python cần thiết
├── runtime.txt              # Đinh dạng phiên bản Python runtime
├── Dockerfile               # File đóng gói môi trường Docker
├── .devcontainer/           # Cấu hình môi trường devcontainer cho VS Code / GitHub Codespaces
└── README.md                # Tài liệu hướng dẫn sử dụng chi tiết
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Yêu cầu tiên quyết
* Python phiên bản **3.10** hoặc **3.11** (khuyên dùng 3.11).
* Khóa Gemini API Key từ [Google AI Studio](https://aistudio.google.com/).

### 2. Tải mã nguồn về máy
```bash
git clone https://github.com/trongquy1811/TimKiemNoiDung.git
cd TimKiemNoiDung
```

### 3. Cài đặt các thư viện phụ thuộc
Khuyên dùng môi trường ảo (`venv`):
```bash
# Tạo và kích hoạt virtual environment
python -m venv venv

# Trên Windows Command Prompt / PowerShell:
.\venv\Scripts\activate

# Trên Linux / macOS:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 4. Cấu hình Gemini API Key
Bạn có thể cấu hình API Key bằng một trong 2 cách sau:
* **Cách 1 (Khuyên dùng):** Nhập trực tiếp API Key vào ô **🔑 Gemini API Key** trên Sidebar giao diện ứng dụng.
* **Cách 2:** Tạo file `.streamlit/secrets.toml` trong thư mục gốc dự án với nội dung:
  ```toml
  GEMINI_API_KEY = "your_gemini_api_key_here"
  ```

### 5. Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Sau khi chạy command trên, trình duyệt sẽ tự động mở trang web ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 💡 Hướng Dẫn Sử Dụng (Usage Guide)

1. **Tải tài liệu / hình ảnh:**
   * Tại vùng **📂 Tải tài liệu hoặc hình ảnh lên**, nhấn hoặc kéo thả các tệp tài liệu (`.pdf`, `.md`, `.txt`) hoặc hình ảnh (`.png`, `.jpg`, `.jpeg`).
   * Nếu bạn tải lên file PDF bị quét (PDF dạng ảnh scanned), ứng dụng sẽ tự động trích xuất ảnh trên từng trang để xem trước và gửi cho Gemini Vision.
2. **Cấu hình thuật toán & mô hình (Tại Sidebar bên trái):**
   * **Chọn mô hình Gemini:** `gemini-2.5-flash` (nhanh & thông minh), `gemini-1.5-pro` (suy luận sâu) hoặc `gemini-2.0-flash-exp`.
   * **Chọn thuật toán RAG:** `Semantic Search (Gemini Embeddings)` hoặc `Keyword Search (Từ khóa Cục bộ)`.
   * Điều chỉnh slider **Độ sáng tạo (Temperature)**, **Giới hạn tokens**, **Số lượng ngữ cảnh (Top-K)** tùy nhu cầu.
3. **Trò chuyện & Trích xuất thông tin:**
   * Nhập câu hỏi vào ô chat phía dưới màn hình (ví dụ: *"Tóm tắt nội dung chính của tài liệu"*, *"Tìm các thông tin về hợp đồng trong file PDF"* hoặc *"Phân tích hình ảnh này giúp tôi"*).
   * AI sẽ trả lời chi tiết dưới dạng Markdown đẹp mắt.
   * Mở phần **📚 Xem context đã trích xuất** dưới câu trả lời của AI để kiểm tra các đoạn văn bản đã được trích xuất từ tài liệu.

---

## 📝 Đóng Góp & Giấy Phép (License & Credits)

Dự án được phát triển nhằm mang lại giải pháp tìm kiếm và phân tích nội dung tài liệu tiếng Việt thông minh, trực quan và mạnh mẽ nhất.

Mọi đóng góp, báo lỗi (issue) hoặc yêu cầu tính năng mới (pull request) đều được hoan nghênh tại repository [TimKiemNoiDung](https://github.com/trongquy1811/TimKiemNoiDung.git).
import streamlit as st
import requests

st.title("Ứng dụng Dịch máy Anh - Việt")
st.subheader("Sử dụng kiến trúc RNN - GRU kết hợp Luong Attention")

# Ô nhập dữ liệu
src_text = st.text_area("Nhập câu tiếng Anh cần dịch:", "Hello, how are you?")

if st.button("Dịch"):
    if src_text.strip() == "":
        st.warning("Vui lòng nhập văn bản!")
    else:
        # Gọi sang API FastAPI (Giả định đang chạy ở cổng 8000)
        try:
            response = requests.post(
                "http://localhost:8000/translate",
                json={"text": src_text}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"Kết quả dịch: {result['translated']}")
            else:
                st.error("Lỗi hệ thống API!")
        except Exception as e:
            st.error(f"Không thể kết nối đến API. Lỗi: {e}")
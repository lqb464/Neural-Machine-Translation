FROM python:3.9-slim

WORKDIR /app

# Copy file cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY ./src ./src
COPY ./deployment/app ./app
COPY ./weights ./weights

# Cổng chạy API
EXPOSE 8000

# Lệnh khởi chạy FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
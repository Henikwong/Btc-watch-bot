FROM python:3.12-slim
# 安装系统证书
RUN apt-get update && apt-get install -y ca-certificates
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "autotrader.py"]

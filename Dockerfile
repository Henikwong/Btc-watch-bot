FROM python:3.12-slim

# 安装证书 & 基础工具
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 启动
CMD ["python", "autotrader.py"]
ENV PYTHONUNBUFFERED=1

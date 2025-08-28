FROM python:3.12-slim

# 安装证书 & 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    update-ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置环境变量 (避免输出缓冲)
ENV PYTHONUNBUFFERED=1

# 启动程序
CMD ["python", "autotrader.py"]

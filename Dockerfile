# 使用 Python 3.12 官方 slim 镜像
FROM python:3.12-slim

# 安装系统证书和依赖
RUN apt-get update && apt-get install -y ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制代码到容器
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 启动脚本
CMD ["python", "autotrader.py"]

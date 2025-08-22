FROM python:3.12-slim

# 安装系统证书
RUN apt-get update && apt-get install -y ca-certificates

# 设置工作目录
WORKDIR /app

# 复制代码
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 启动脚本
CMD ["python", "autotrader.py"]

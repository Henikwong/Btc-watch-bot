# 基础镜像
FROM python:3.12-slim

# 安装系统依赖和证书
RUN apt-get update && \
    apt-get install -y ca-certificates build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制代码
COPY . .

# 使用国内镜像安装 Python 依赖，加快速度
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 启动脚本
CMD ["python", "autotrader.py"]

FROM python:3.12-slim

# 安装系统依赖（编译 numpy/ta 等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    libffi-dev \
    libfreetype6-dev \
    pkg-config \
    ca-certificates \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 升级 pip 并安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认命令（可根据你的项目改）
CMD ["python", "main.py"]

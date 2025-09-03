FROM python:3.12-slim

# 安装证书 & 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential gcc g++ gfortran libssl-dev libffi-dev libatlas-base-dev && \
    update-ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN useradd -m botuser
USER botuser

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 删除编译工具，瘦身镜像
USER root
RUN apt-get purge -y build-essential gcc g++ gfortran libatlas-base-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 切回非 root 用户
USER botuser

# 复制代码
COPY --chown=botuser:botuser . .

# 挂载配置和日志目录
VOLUME /app/config
VOLUME /app/logs

# 设置环境变量 (避免输出缓冲)
ENV PYTHONUNBUFFERED=1
ENV APP_CONFIG=/app/config/config.yaml
ENV LOG_DIR=/app/logs

# 启动程序
CMD ["python", "autotrader.py"]

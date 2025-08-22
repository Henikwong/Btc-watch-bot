# 使用一个包含Python 3.11的官方Docker镜像作为基础镜像。
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 文件到容器中
COPY requirements.txt.

# 使用 pip 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 复制您的所有项目文件到容器中
COPY..

# 定义容器启动时要执行的命令
CMD ["python", "autotreader.py"]

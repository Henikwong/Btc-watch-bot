# 使用一个包含Python 3.11的官方Docker镜像作为基础镜像。
# 这可以避免因Python版本不匹配导致的部署错误 [5]。
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 文件到容器中
# 注意: 文件名和点之间有一个空格，这是正确的语法 [4]。
COPY requirements.txt.

# 使用 pip 安装依赖项
# --no-cache-dir 参数可以减小镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 复制您的所有项目文件到容器中
# 同样，文件名和点之间有一个空格 [4]。
COPY..

# 定义容器启动时要执行的命令
# 您的交易脚本是一个长时运行的后台进程 [4, 6, 3]，因此不需要公开端口。
# CMD指令会启动您的脚本。
CMD ["python", "autotreader.py"]

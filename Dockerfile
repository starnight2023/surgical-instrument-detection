# 使用官方 PyTorch 基础镜像，默认支持 CUDA GPU 加速
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 设置环境变量，防止 Python 缓冲输出和生成编译文件
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装 OpenCV 核心依赖及常用工具（关键，否则 opencv 导入会报错 libGL）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖声明并进行安装
COPY requirements.txt .

# 如果在中国大陆地区构建，可以取消下面这一行的注释来配置清华源镜像加速安装
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 切换为腾讯云官方 PyPI 镜像源，并将其设为信任域名
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.cloud.tencent.com

RUN pip install --no-cache-dir -r requirements.txt

# 将剩余的所有项目源码复制到工作目录中（大体积数据已被 .dockerignore 排除）
COPY . .


# 暴露端口
EXPOSE 8000
EXPOSE 8501

# 默认启动命令（如果直接运行 Dockerfile，默认启动 Streamlit）
CMD ["streamlit", "run", "streamlit_Pre.py", "--server.address=0.0.0.0", "--server.port=8501"]
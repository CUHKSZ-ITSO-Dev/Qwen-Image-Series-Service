# 使用体积小一点的pytorch镜像
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# 设置工作目录
WORKDIR /app

RUN DEBIAN_FRONTEND=noninteractive apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y curl git tzdata vim

# 设置时区为Asia/Shanghai
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 复制项目配置文件
COPY pyproject.toml uv.lock* ./

# 安装Python依赖
RUN /root/.local/bin/uv sync --no-dev
RUN find / -name "serve"

# 复制应用代码
COPY main.py ./

# 暴露端口
EXPOSE 8000

# 健康检查
# HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# 设置默认环境变量
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

# 启动命令 - 指向新的 deployment_graph 对象
# CMD ["/root/.local/bin/serve", "run", "main:deployment_graph"]

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

# 复制应用代码
COPY main.py service.py ./

# 暴露端口
EXPOSE 8000

# 设置默认环境变量
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

LABEL org.opencontainers.image.description="Qwen 图像系列服务\n\n目前市面上缺乏为 Diffusion 类模型提供 OpenAI 兼容接口的 Web 框架或服务器，因此本项目对 Qwen-Image 和 Qwen-Image-Edit 进行了简单封装，实现了兼容 OpenAI 接口的图像生成与编辑服务。\n\n底层使用 HuggingFace 的 Diffusers 库，利用 Pipeline 实现 Qwen 模型的加载和推理生成。\n\nWeb 服务器选用 FastAPI。\n\n使用 Ray Server 为模型 Batch 请求批处理预备好。*\n\n*: 由于批处理局限性太大，以及当前算力资源有限，Batch 并没有什么优势，故现在没有启用 Ray Server 的 Batch 模式，仍然保持线性同步处理。\n\n## 运行服务\n\n您可以使用 Docker 来运行此服务：\n\n    docker build -t qwen-image-service .\n    docker run -p 8000:8000 --gpus all qwen-image-service\n\n服务将在 http://localhost:8000 上可用。\n\n目前为一个模型分了一张 GPU，可以视情况修改代码。\n\n## 环境变量配置\n\n服务支持以下环境变量进行配置：\n\n| 环境变量 | 默认值 | 描述 |\n| -------- | ------ | ---- |\n| QWEN_IMAGE_EDIT_LOCATION | /qwen-image-edit | Qwen-Image-Edit 模型路径 |\n| QWEN_IMAGE_LOCATION | /qwen-image | Qwen-Image 生成模型路径 |\n\n你可以配置为 Qwen/Qwen-Image-Edit 等自动从 HuggingFace 上面下载模型并加载。\n\n## API 文档\n\n详见项目 README.md。"
LABEL org.opencontainers.image.title="Qwen 图像系列服务"
LABEL org.opencontainers.image.source="https://github.com/cuhksz-itso-dev/qwen-image-series-service"
LABEL org.opencontainers.image.url="https://github.com/cuhksz-itso-dev/qwen-image-series-service/pkgs/container/qwen-image-series-svc"
LABEL org.opencontainers.image.documentation="https://github.com/cuhksz-itso-dev/qwen-image-series-service/blob/main/README.md"

# 启动命令
CMD ["/app/.venv/bin/serve", "run", "main:deployment_graph"]

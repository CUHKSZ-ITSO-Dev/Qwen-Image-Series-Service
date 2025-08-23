import asyncio
import base64
import gc
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, List

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from diffusers import QwenImageEditPipeline

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型
pipeline = None
device = None


# 响应模型
class ImageEditResponse(BaseModel):
    """图像编辑响应模型"""

    created: int
    data: List[dict]


class ImageData(BaseModel):
    """图像数据模型"""

    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


# 模型加载和清理的生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global pipeline, device

    logger.info("正在加载Qwen图像编辑模型...")
    try:
        # 检查CUDA可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")

        dtype = torch.bfloat16

        pipeline = QwenImageEditPipeline.from_pretrained(
            "/model",
            torch_dtype=dtype
        )
        logger.info("模型加载完成！")
        yield

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise
    finally:
        # 清理资源
        if pipeline is not None:
            del pipeline

        # 强制垃圾回收

        gc.collect()

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("资源清理完成")


# 创建FastAPI应用
app = FastAPI(
    title="Qwen图像编辑服务",
    description="基于Qwen模型的图像编辑API服务，兼容OpenAI格式",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image(image_file: UploadFile) -> Image.Image:
    """验证并处理上传的图像文件"""
    try:
        # 检查文件类型
        if not image_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必须是图像格式")

        # 读取并转换图像
        image_bytes = image_file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 重置文件指针（如果需要再次读取）
        image_file.file.seek(0)

        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像处理失败: {str(e)}")


def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64


async def process_image_edit(
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """异步处理图像编辑"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 准备生成器
        generator = torch.manual_seed(seed if seed is not None else int(time.time()))

        # 准备输入参数
        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }

        # 在线程池中运行模型推理
        loop = asyncio.get_event_loop()

        def run_inference():
            with torch.inference_mode():
                # 清理GPU缓存在推理前
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                output = pipeline(**inputs)
                result_image = output.images[0]

                # 清理GPU缓存在推理后
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return result_image

        # 异步执行
        output_image = await loop.run_in_executor(None, run_inference)

        # 强制垃圾回收

        gc.collect()

        return output_image

    except Exception as e:
        logger.error(f"图像编辑处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"图像编辑失败: {str(e)}")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": "Qwen图像编辑服务正在运行",
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": device,
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": device,
        "timestamp": int(time.time()),
    }


@app.post("/v1/images/edits", response_model=ImageEditResponse)
async def create_image_edit(
    image: UploadFile = File(..., description="要编辑的原始图像"),
    prompt: str = Form(..., description="图像编辑提示词"),
    negative_prompt: str = Form("", description="负面提示词"),
    num_inference_steps: int = Form(50, description="推理步数"),
    true_cfg_scale: float = Form(4.0, description="CFG尺度"),
    seed: Optional[int] = Form(None, description="随机种子"),
    response_format: str = Form("b64_json", description="响应格式: b64_json 或 url"),
):
    """
    图像编辑API - 兼容OpenAI格式

    根据提示词编辑上传的图像
    """
    try:
        # 验证输入参数
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="提示词不能为空")

        if num_inference_steps < 1 or num_inference_steps > 100:
            raise HTTPException(status_code=400, detail="推理步数必须在1-100之间")

        if true_cfg_scale < 0.1 or true_cfg_scale > 20.0:
            raise HTTPException(status_code=400, detail="CFG尺度必须在0.1-20.0之间")

        # 验证和处理图像
        input_image = validate_image(image)

        logger.info(
            f"开始处理图像编辑请求: prompt='{prompt}', steps={num_inference_steps}"
        )

        # 异步处理图像编辑
        output_image = await process_image_edit(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
        )

        # 准备响应数据
        if response_format == "b64_json":
            image_b64 = image_to_base64(output_image)
            image_data = {"b64_json": image_b64}
        else:
            # 注意：这里需要实现图像存储和URL生成逻辑
            raise HTTPException(status_code=400, detail="当前仅支持b64_json格式")

        response = ImageEditResponse(created=int(time.time()), data=[image_data])

        logger.info("图像编辑请求处理完成")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时发生未预期的错误: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")


if __name__ == "__main__":
    # 从环境变量获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))

    logger.info(f"启动服务器: http://{host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        loop="asyncio",
        log_level="info",
    )

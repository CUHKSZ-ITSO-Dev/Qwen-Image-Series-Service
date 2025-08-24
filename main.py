import base64
import io
import time
from typing import Annotated

import torch
from diffusers import DiffusionPipeline, QwenImageEditPipeline
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from ray import serve


class BaseService:
    """封装通用批处理逻辑的基类"""

    def _prepare_request(self, request: dict) -> dict:
        """子类需要实现此方法，以准备单个请求的参数字典。"""
        raise NotImplementedError

    async def _process_batch(self, requests: list[dict]) -> list[Image.Image]:
        """通用的批处理流程。"""
        if not requests:
            return []

        # 1. 使用子类实现的 _prepare_request 方法来准备所有请求
        full_requests = [self._prepare_request(r) for r in requests]

        # 2. 将字典列表转换为列表字典
        keys = full_requests[0].keys()
        batch_params = {key: [d[key] for d in full_requests] for key in keys}

        # 3. 调用 self.pipeline (由子类提供)
        with torch.inference_mode():
            model_output = self.pipeline(**batch_params)
            return model_output.images


@serve.deployment(ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0})
class ImageEditService(BaseService):
    """专门用于图像编辑的模型服务：Qwen-Image-Edit"""

    def __init__(self) -> None:
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            "/qwen-image-edit", torch_dtype=torch.bfloat16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(self.device)

    def _prepare_request(self, r: dict) -> dict:
        seed = r.get("seed", 42)
        return {
            "image": r["image"],
            "prompt": r["prompt"],
            "negative_prompt": r.get("negative_prompt", " "),
            "num_inference_steps": r.get("num_inference_steps", 50),
            "true_cfg_scale": r.get("true_cfg_scale", 4.0),
            "generator": torch.Generator(device=self.device).manual_seed(seed),
        }

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=8)
    async def batch_edit(self, requests: list[dict]) -> list[Image.Image]:
        return await self._process_batch(requests)


@serve.deployment(ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0})
class ImageGenerationService(BaseService):
    """专门用于图像生成的模型服务：Qwen-Image"""

    def __init__(self) -> None:
        self.pipeline = DiffusionPipeline.from_pretrained(
            "/qwen-image", torch_dtype=torch.bfloat16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(self.device)

    def _prepare_request(self, r: dict) -> dict:
        seed = r.get("seed", 42)
        return {
            "prompt": r["prompt"],
            "negative_prompt": r.get("negative_prompt", ""),
            "num_inference_steps": r.get("num_inference_steps", 50),
            "width": r.get("width", 1664),
            "height": r.get("height", 928),
            "true_cfg_scale": r.get("true_cfg_scale", 4.0),
            "generator": torch.Generator(device=self.device).manual_seed(seed),
        }

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=8)
    async def batch_generate(self, requests: list[dict]) -> list[Image.Image]:
        return await self._process_batch(requests)


class EditRequest(BaseModel):
    """图像编辑请求的表单字段"""

    prompt: str
    negative_prompt: str | None = None
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None
    seed: int | None = None


class GenerationRequest(BaseModel):
    """图像生成请求的表单字段"""

    prompt: str
    width: int | None = None
    height: int | None = None
    negative_prompt: str | None = None
    num_inference_steps: int | None = 50
    true_cfg_scale: float | None = 4.0
    seed: int | None = 42


app = FastAPI()
edit_deployment = ImageEditService.bind()
generation_deployment = ImageGenerationService.bind()


def form_body(
    prompt: Annotated[str, Form(...)],
    negative_prompt: Annotated[str | None, Form(None)] = None,
    num_inference_steps: Annotated[int | None, Form(None)] = None,
    true_cfg_scale: Annotated[float | None, Form(None)] = None,
    seed: Annotated[int | None, Form(None)] = None,
) -> EditRequest:
    """依赖函数，将 Form 字段解析并校验为 Pydantic 模型"""
    return EditRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        seed=seed,
    )


@app.post("/v1/images/edits")
async def edit_image(
    image: Annotated[UploadFile, File(...)],
    request: Annotated[EditRequest, Depends(form_body)],
) -> JSONResponse:
    """图像编辑端点"""
    init_image = Image.open(io.BytesIO(await image.read())).convert("RGB")

    # 从 Pydantic 模型创建载荷，exclude_none=True 会自动过滤掉未提供的可选参数
    payload = request.model_dump(exclude_none=True)
    payload["image"] = init_image

    result_image = await edit_deployment.batch_edit.remote(payload)

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    b64_json = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse(
        content={"created": int(time.time()), "data": [{"b64_json": b64_json}]}
    )


@app.post("/v1/images/generations")
async def generate_image(request: GenerationRequest) -> JSONResponse:
    """图像生成端点"""
    request_dict = request.model_dump(exclude_none=True)

    result_image = await generation_deployment.batch_generate.remote(request_dict)

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    b64_json = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse(
        content={"created": int(time.time()), "data": [{"b64_json": b64_json}]}
    )


deployment_graph = serve.ingress(app)

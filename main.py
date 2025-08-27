import base64
import io
import time
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle

from service import ImageEditService, ImageGenerationService

app = FastAPI()


class GenerationRequest(BaseModel):
    """图像生成请求的表单字段"""

    prompt: str
    width: int | None = None
    height: int | None = None
    negative_prompt: str | None = None
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None
    seed: int | None = None


class EditRequest(BaseModel):
    """图像编辑请求的表单字段"""

    prompt: str
    negative_prompt: str | None = None
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None
    seed: int | None = None


def edit_form_body(
    prompt: Annotated[str, Form(...)],
    negative_prompt: Annotated[str | None, Form()] = None,
    num_inference_steps: Annotated[int | None, Form()] = None,
    true_cfg_scale: Annotated[float | None, Form()] = None,
    seed: Annotated[int | None, Form()] = None,
) -> EditRequest:
    """依赖函数，将 Form 字段解析并校验为 Pydantic 模型"""
    return EditRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        seed=seed,
    )


@serve.deployment
@serve.ingress(app)
class APIIngress:
    def __init__(
        self,
        edit_deployment: DeploymentHandle,
        generation_deployment: DeploymentHandle,
    ) -> None:
        self.edit_deployment = edit_deployment
        self.generation_deployment = generation_deployment

    @app.post("/v1/images/edits")
    async def edit_image(
        self,
        image: Annotated[UploadFile, File(...)],
        request: Annotated[EditRequest, Depends(edit_form_body)],
    ) -> JSONResponse:
        """图像编辑端点"""
        init_image = Image.open(io.BytesIO(await image.read())).convert("RGB")

        # 从 Pydantic 模型创建载荷，exclude_none=True 会自动过滤掉未提供的可选参数
        payload = request.model_dump(exclude_none=True)
        payload["image"] = init_image

        result_image = await self.edit_deployment.edit.remote(payload)

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        b64_json = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(
            content={"created": int(time.time()), "data": [{"b64_json": b64_json}]}
        )

    @app.post("/v1/images/generations")
    async def generate_image(self, request: GenerationRequest) -> JSONResponse:
        """图像生成端点"""
        request_dict = request.model_dump(exclude_none=True)

        result_image = await self.generation_deployment.generate.remote(request_dict)

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        b64_json = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(
            content={"created": int(time.time()), "data": [{"b64_json": b64_json}]}
        )


deployment_graph = APIIngress.bind(
    edit_deployment=ImageEditService.bind(),
    generation_deployment=ImageGenerationService.bind(),
)

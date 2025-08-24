import io
from typing import Annotated

import torch
from diffusers import QwenImageEditPipeline
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from ray import serve

app = FastAPI()


@serve.deployment(
    ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0},
    max_concurrent_queries=16,
)
@serve.ingress(app)
class QwenImageEditService:
    def __init__(self) -> None:
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            "/model", torch_dtype=torch.bfloat16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(self.device)
        self.default_params = {
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def batch_generate(self, requests: list[dict]) -> list[Image.Image]:
        """
        批处理请求。由 FastAPI 内部发起。

        这个方法会接收一批请求，每个请求都是一个字典。
        然后它会解包这些数据，以列表形式传递给 pipeline。
        """
        images = [r["image"] for r in requests]
        prompts = [r["prompt"] for r in requests]
        with torch.inference_mode():
            model_output = self.pipeline(
                image=images, prompt=prompts, **self.default_params
            )
            return model_output.images

    @app.post("/")
    async def generate(
        self,
        image: Annotated[UploadFile, File(...)],
        prompt: Annotated[str, Form("")],
    ) -> StreamingResponse:
        """
        FastAPI 端点。

        它会将请求数据预处理后，调用批处理方法进行推理。
        """
        init_image = Image.open(io.BytesIO(await image.read())).convert("RGB")

        # 异步调用批处理方法
        result_image = await self.batch_generate.remote(
            {"image": init_image, "prompt": prompt}
        )

        # 将生成的图片转换为流式响应
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")


# 绑定 Ray Serve deployment
qwen_image_edit_service = QwenImageEditService.bind()

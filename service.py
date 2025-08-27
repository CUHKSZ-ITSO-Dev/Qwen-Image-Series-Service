import torch
from diffusers import QwenImageEditPipeline, QwenImagePipeline
from PIL import Image
from ray import serve


class _BaseService:
    """封装通用单请求推理逻辑的基类"""

    def _prepare_request(self, request: dict) -> dict:
        """子类需要实现此方法，以准备单个请求的参数字典。"""
        raise NotImplementedError

    async def _process_single(self, request: dict) -> Image.Image:
        """通用的单请求推理流程。"""
        params = self._prepare_request(request)
        with torch.inference_mode():
            model_output = self.pipeline(**params)
            return model_output.images[0]


@serve.deployment(ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0})
class ImageEditService(_BaseService):
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
            "num_inference_steps": r.get("num_inference_steps", 40),
            "true_cfg_scale": r.get("true_cfg_scale", 4.0),
            "generator": torch.Generator(device=self.device).manual_seed(seed),
        }

    async def edit(self, request: dict) -> Image.Image:
        return await self._process_single(request)


@serve.deployment(ray_actor_options={"num_gpus": 1 if torch.cuda.is_available() else 0})
class ImageGenerationService(_BaseService):
    """专门用于图像生成的模型服务：Qwen-Image"""

    def __init__(self) -> None:
        self.pipeline = QwenImagePipeline.from_pretrained(
            "/qwen-image", torch_dtype=torch.bfloat16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline.to(self.device)

    def _prepare_request(self, r: dict) -> dict:
        seed = r.get("seed", 42)
        return {
            "prompt": r["prompt"],
            "negative_prompt": r.get("negative_prompt", ""),
            "num_inference_steps": r.get("num_inference_steps", 40),
            "width": r.get("width", 1664),
            "height": r.get("height", 928),
            "true_cfg_scale": r.get("true_cfg_scale", 4.0),
            "generator": torch.Generator(device=self.device).manual_seed(seed),
        }

    async def generate(self, request: dict) -> Image.Image:
        return await self._process_single(request)


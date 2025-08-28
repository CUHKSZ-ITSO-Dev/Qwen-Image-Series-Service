# Qwen 图像系列服务

[English](README_EN.md) | 中文

---

目前市面上缺乏为 Diffusion 类模型提供 OpenAI 兼容接口的 Web 框架或服务器，因此本项目对 Qwen-Image 和 Qwen-Image-Edit 进行了简单封装，实现了兼容 OpenAI 接口的图像生成与编辑服务。

底层使用 HuggingFace 的 Diffusers 库，利用 Pipeline 实现 Qwen 模型的加载和推理生成。

Web 服务器选用 FastAPI。

使用 Ray Server 为模型 Batch 请求批处理预备好。*

*: 由于批处理局限性太大，以及当前算力资源有限，Batch 并没有什么优势，故现在没有启用 Ray Server 的 Batch 模式，仍然保持线性同步处理。

## 运行服务

您可以使用 Docker 来运行此服务。我预构建了 linux/amd64 架构的镜像，你可以直接从 Package 中拉取并运行：
```bash
docker run -p 8000:8000 --gpus all ghcr.io/cuhksz-itso-dev/qwen-image-series-svc:latest
```

也可以本地构建镜像并运行：
```bash
docker build -t qwen-image-series-svc .
docker run -p 8000:8000 --gpus all qwen-image-series-svc
```

服务将在 `http://localhost:8000` 上可用。

目前为一个模型分了一张 GPU，可以视情况修改代码。

## 环境变量配置

服务支持以下环境变量进行配置：

| 环境变量 | 默认值 | 描述 |
| -------- | ------ | ---- |
| `QWEN_IMAGE_EDIT_LOCATION` | `/qwen-image-edit` | Qwen-Image-Edit 模型路径 |
| `QWEN_IMAGE_LOCATION` | `/qwen-image` | Qwen-Image 生成模型路径 |

- 你可以配置为 `Qwen/Qwen-Image-Edit` 等自动从 HuggingFace 上面下载模型并加载。

### 使用环境变量运行

```bash
# 自定义端口和模型路径
docker run -p 8000:8000 \
  -e MODEL_PATH_EDIT=/custom/path/to/qwen-image-edit \
  -e MODEL_PATH_GENERATION=/custom/path/to/qwen-image \
  --gpus all qwen-image-series-svc
```

## API 文档

服务提供两个核心端点，均与 OpenAI API 规范兼容。

---

### 1. 图像生成

根据文本提示生成一张新图像，类似于 DALL-E。

- **端点:** `/v1/images/generations`
- **方法:** `POST`
- **请求体格式:** `application/json`

#### 请求参数

| 字段                  | 类型    | 是否必须 | 默认值   | 描述                                     |
| --------------------- | ------- | -------- | -------- | ---------------------------------------- |
| `prompt`              | 字符串  | 是       | -        | 对期望图像的文本描述。                   |
| `width`               | 整数    | 否       | `1664`   | 生成图像的宽度。                         |
| `height`              | 整数    | 否       | `928`    | 生成图像的高度。                         |
| `negative_prompt`     | 字符串  | 否       | `""`     | 描述图像中应避免出现元素的文本。         |
| `num_inference_steps` | 整数    | 否       | `50`     | 推理步数。                               |
| `true_cfg_scale`      | 浮点数  | 否       | `4.0`    | 无分类器指导（CFG）的缩放因子。          |
| `seed`                | 整数    | 否       | `42`     | 用于随机数生成的种子。                   |

#### 请求示例 (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/v1/images/generations" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "一只可爱的猫在沙发上睡觉"
         }'
```

---

### 2. 图像编辑

根据给定的图像和文本提示，创建编辑后或变体的图像。

- **端点:** `/v1/images/edits`
- **方法:** `POST`
- **请求体格式:** `multipart/form-data`

#### 请求参数

| 字段                  | 类型    | 是否必须 | 默认值   | 描述                                     |
| --------------------- | ------- | -------- | -------- | ---------------------------------------- |
| `image`               | 文件    | 是       | -        | 需要编辑的图像。必须是有效的图像文件（如 PNG, JPEG 等）。  |
| `prompt`              | 字符串  | 是       | -        | 对期望编辑效果的文本描述。               |
| `negative_prompt`     | 字符串  | 否       | `" "`    | 描述图像中应避免出现元素的文本。         |
| `num_inference_steps` | 整数    | 否       | `50`     | 推理步数。                               |
| `true_cfg_scale`      | 浮点数  | 否       | `4.0`    | 无分类器指导（CFG）的缩放因子。          |
| `seed`                | 整数    | 否       | `42`     | 用于随机数生成的种子。                   |

#### 请求示例 (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/v1/images/edits" \
     -F "image=@/path/to/your/image.png" \
     -F "prompt="给猫戴上派对帽""
```

---

### 成功响应 (适用于所有端点)

- **状态码:** `200 OK`
- **响应体:**

返回一个 JSON 对象，包含创建时间戳和生成的图像数据。

```json
{
  "created": 1686663552,
  "data": [
    {
      "b64_json": "<图像的 base64 编码字符串>"
    }
  ]
}
```

### 错误响应

如果请求无效，API 将返回 `4xx` 状态码，并在 JSON 响应体中描述错误信息。

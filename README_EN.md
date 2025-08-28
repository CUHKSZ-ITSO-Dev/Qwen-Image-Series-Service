# Qwen Image Series Service

English | [中文](README.md)

---

Currently, there is a lack of web frameworks or servers that provide OpenAI-compatible interfaces for Diffusion models. Therefore, this project provides a simple wrapper for Qwen-Image and Qwen-Image-Edit, implementing image generation and editing services compatible with OpenAI's interface.

The underlying implementation uses HuggingFace's Diffusers library, utilizing Pipeline to load and perform inference with Qwen models.

FastAPI is chosen as the web server.

Ray Server is prepared for model batch request processing.*

*: Due to the significant limitations of batch processing and limited current computing resources, batch processing doesn't provide much advantage, so Ray Server's batch mode is not currently enabled, maintaining linear synchronous processing instead.

## Running the Service

You can use Docker to run this service. I have pre-built the image for the linux/amd64 architecture, which you can directly pull from the package and run:
```bash
docker run -p 8000:8000 --gpus all ghcr.io/cuhksz-itso-dev/qwen-image-series-svc:latest
```

You can also build and run the image locally:
```bash
docker build -t qwen-image-series-svc .
docker run -p 8000:8000 --gpus all qwen-image-series-svc
```

The service will be available at `http://localhost:8000`.

Currently, one GPU is allocated per model, which can be modified in the code as needed.

## Environment Variables Configuration

The service supports the following environment variables for configuration:

| Environment Variable | Default Value | Description |
| ------------------- | ------------- | ----------- |
| `QWEN_IMAGE_EDIT_LOCATION` | `/qwen-image-edit` | Qwen-Image-Edit model path |
| `QWEN_IMAGE_LOCATION` | `/qwen-image` | Qwen-Image generation model path |

- You can configure it as `Qwen/Qwen-Image-Edit` to automatically download and load models from HuggingFace.

### Running with Environment Variables

```bash
# Custom port and model paths
docker run -p 8000:8000 \
  -e MODEL_PATH_EDIT=/custom/path/to/qwen-image-edit \
  -e MODEL_PATH_GENERATION=/custom/path/to/qwen-image \
  --gpus all qwen-image-series-svc
```

## API Documentation

The service provides two core endpoints, both compatible with the OpenAI API specification.

---

### 1. Image Generation

Generates a new image from a text prompt, similar to DALL-E.

- **Endpoint:** `/v1/images/generations`
- **Method:** `POST`
- **Request Body:** `application/json`

#### Request Parameters

| Field                 | Type    | Required | Default | Description                                            |
| --------------------- | ------- | -------- | ------- | ------------------------------------------------------ |
| `prompt`              | string  | Yes      | -       | A text description of the desired image(s).            |
| `width`               | integer | No       | `1664`  | The width of the generated image.                      |
| `height`              | integer | No       | `928`   | The height of the generated image.                     |
| `negative_prompt`     | string  | No       | `""`    | A text description of elements to avoid in the image.  |
| `num_inference_steps` | integer | No       | `50`    | The number of denoising steps.                         |
| `true_cfg_scale`      | float   | No       | `4.0`   | Classifier-Free Guidance scale.                        |
| `seed`                | integer | No       | `42`    | The seed for random number generation.                 |

#### Example Request (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/v1/images/generations" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "A cute cat sleeping on a sofa"
         }'
```

---

### 2. Image Editing

Creates an edited or variation of a given image based on a text prompt.

- **Endpoint:** `/v1/images/edits`
- **Method:** `POST`
- **Request Body:** `multipart/form-data`

#### Request Parameters

| Field                 | Type    | Required | Default | Description                                            |
| --------------------- | ------- | -------- | ------- | ------------------------------------------------------ |
| `image`               | file    | Yes      | -       | The image to edit. Must be a valid image file (e.g., PNG, JPEG). |
| `prompt`              | string  | Yes      | -       | A text description of the desired edits.               |
| `negative_prompt`     | string  | No       | `" "`   | A text description of elements to avoid in the image.  |
| `num_inference_steps` | integer | No       | `50`    | The number of denoising steps.                         |
| `true_cfg_scale`      | float   | No       | `4.0`   | Classifier-Free Guidance scale.                        |
| `seed`                | integer | No       | `42`    | The seed for random number generation.                 |

#### Example Request (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/v1/images/edits" \
     -F "image=@/path/to/your/image.png" \
     -F "prompt="make the cat wear a party hat""
```

---

### Success Response (for both endpoints)

- **Status Code:** `200 OK`
- **Body:**

Returns a JSON object containing the timestamp of creation and the generated image data.

```json
{
  "created": 1686663552,
  "data": [
    {
      "b64_json": "<base64-encoded-string-of-the-image>"
    }
  ]
}
```

### Error Response

If the request is invalid, the API will return an error with a `4xx` status code and a JSON body describing the error.

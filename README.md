# Text-to-Image-by-Stable-Diffusion
# Stable Diffusion Image Generation with Transformers

This repository contains code for generating images using Stable Diffusion, powered by Hugging Face Transformers and the Diffusers library. The Stable Diffusion pipeline allows you to create high-quality and diverse images based on textual prompts.

## Getting Started

To get started, you'll need to have the following dependencies installed:

- NVIDIA GPU: Make sure you have an NVIDIA GPU as this code is optimized for GPU acceleration.

### Installation

You can install the necessary Python packages using `pip`:

```bash
!pip install diffusers==0.11.1
!pip install transformers scipy ftfy accelerate
```

## Model Selection

There are various Stable Diffusion models available for generating images. Here are some options you can explore:

- [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) (supports 768x768 resolution)

We recommend loading models from the half-precision branch [`fp16`](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/fp16) for compatibility with free Google Colab instances. Additionally, inform the `diffusers` library to expect weights in float16 precision by passing `torch_dtype=torch.float16`.

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

## GPU Acceleration

For faster inference, move the pipeline to the GPU:

```python
pipe = pipe.to("cuda")
```

## Image Generation

You can generate images by providing a textual prompt:

```python
prompt = "Photograph of a busy city street at dusk, taken from a low angle, with towering buildings and vibrant neon lights."
image = pipe(prompt).images[0]
image
```

Running the image generation cell multiple times will produce different images. For deterministic output, you can pass a random seed to the pipeline:

```python
generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, generator=generator).images[0]
image
```

You can also control the number of inference steps for image generation:

```python
generator = torch.Generator("cuda").manual_seed(1024)
image = pipe(prompt, num_inference_steps=15, generator=generator).images[0]
image
```

## Grid Image Display

To display a grid of generated images, you can use the `image_grid` function provided in the README:

```python
from PIL import Image

def image_grid(imgs, rows, cols):
    # Function code here...

# Generate a grid of images
num_images = 3
prompt = ["Prompt 1", "Prompt 2", "Prompt 3"] * num_images
images = pipe(prompt).images
grid = image_grid(images, rows=1, cols=3)
grid
```

You can adapt this function to display grids with different numbers of rows and columns.

## Generating Non-Square Images

Stable Diffusion typically produces images of `512 × 512` pixels. You can easily create rectangular images in portrait or landscape ratios using the `height` and `width` arguments. Some recommendations:

- Ensure both `height` and `width` are multiples of `8`.
- Going below 512 might result in lower quality images.
- Going above 512 in both directions can result in repeated image areas.

```python
prompt = "Long-exposure photograph of a busy intersection at night, capturing the streaks of light from moving vehicles"
image = pipe(prompt, height=512, width=768).images[0]
image
```

Have fun exploring Stable Diffusion and creating stunning images with textual prompts! If you have any questions or need assistance, feel free to reach out.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="DALL-E Image Generation Service")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

client = OpenAI()  # uses env var


class ImageRequest(BaseModel):
    prompt: str
    model: str | None = "gpt-image-1"   # DALL·E-style model
    size: str | None = "512x512"       # smaller = lighter
    n: int | None = 1                  # one image at a time


class ImageResponse(BaseModel):
    urls: list[str]


@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(body: ImageRequest):
    try:
        result = client.images.generate(
            model=body.model,
            prompt=body.prompt,
            size=body.size,
            n=body.n,
        )
        urls = [d.url for d in result.data]
        return ImageResponse(urls=urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

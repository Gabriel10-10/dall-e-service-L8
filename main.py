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
    model: str | None = "dall-e-3"   # DALL·E-style model
    size: str | None = "1024x1024"       # smaller = lighter
    n: int | None = 1                  # one image at a time


class ImageResponse(BaseModel):
    urls: list[str]


@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(body: ImageRequest):
    try:

        size = body.size or "1024x1024"
        result = client.images.generate(
            model=body.model or "dall-e-3",
            prompt=body.prompt,
            size=size,
            n=body.n or 1,
        )
        urls = [d.url for d in result.data]
        if not urls:
            raise HTTPException(status_code=500, detail="No image URLs returned from OpenAI")
        return ImageResponse(urls=urls)
    except Exception as e:
        print(f"Error in dalle-service: {e}")
        raise HTTPException(status_code=500, detail=str(e))
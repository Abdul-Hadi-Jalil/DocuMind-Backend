from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import base64
from pydantic import BaseModel
from typing import Optional

# modules to convert bytes into image
from PIL import Image
import io
import numpy as np
import easyocr
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Initialize EasyOCR reader once (it's heavy to load)
reader = easyocr.Reader(['en'])

# Initialize OpenAI client for Nebius
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key="v1.CmQKHHN0YXRpY2tleS1lMDB5eTYydHdzeDd5cnZkNWESIXNlcnZpY2VhY2NvdW50LWUwMGE0cWJ3cXFjam15M2FuNTIMCMvqtckGEJ3noeIBOgwIyu3NlAcQwPvd7QJAAloDZTAw.AAAAAAAAAAGrE-gzBauL4i5ofs04Tm_ElU2mTyH8xCn9W5cIyCiYWSFKi3Hoi_g6GYKEBHreqRrnvC3-JTvQrumjKCzvJkIH"
)

# Define request model for generate-image endpoint
class GenerateImageRequest(BaseModel):
    prompt: str
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = 28
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = -1

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image"""
    return Image.open(io.BytesIO(image_bytes))

def image_to_text(image):
    """Convert image to text using EasyOCR"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Perform OCR
    results = reader.readtext(image_np)
    
    # Extract and combine all detected text
    extracted_text = ' '.join([result[1] for result in results])
    return extracted_text

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Convert bytes to image
    image = bytes_to_image(contents)

    # Extracting the text from image
    extracted_text = image_to_text(image)

    print(f"Extracted text: {extracted_text}")
    print(f"Image size: {image.size}")
    
    return {
        "content_type": file.content_type,
        "extracted_text": extracted_text
    }

@app.post("/generate-image/")
async def generate_image(request: GenerateImageRequest):
    """
    Generate an image based on a text prompt using Nebius AI.
    
    Args:
        request: GenerateImageRequest containing prompt and optional parameters
    
    Returns:
        dict: Contains base64 encoded image and metadata
    """
    try:
        # Generate image using Nebius AI
        response = client.images.generate(
            model="black-forest-labs/flux-dev",
            response_format="b64_json",
            extra_body={
                "response_extension": "png",
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "loras": None
            },
            prompt=request.prompt
        )
        
        # Extract the base64 image data
        if hasattr(response, 'data') and len(response.data) > 0:
            image_data = response.data[0].b64_json
            
            # You can also decode the base64 to verify it's valid
            try:
                # Decode base64 to bytes (optional, just for validation)
                image_bytes = base64.b64decode(image_data)
                
                return {
                    "image_b64": image_data,
                }
            except base64.binascii.Error as e:
                raise HTTPException(status_code=500, detail=f"Invalid base64 data: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="No image data received from AI service")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
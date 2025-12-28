from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import base64
from pydantic import BaseModel
from typing import Optional
import ollama

# modules to convert bytes into image
from PIL import Image
import io
import numpy as np
import requests
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

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
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        print(f"Ollama server check: Status {response.status_code}")
        if response.status_code == 200:
            return True, "Ollama server is running"
        else:
            print(f"Ollama server returned non-200 status: {response.status_code}")
            return False, f"Ollama server returned status {response.status_code}"
    except requests.exceptions.ConnectionError as e:
        print(f"Ollama server connection error: {e}")
        return False, f"Connection error: {e}"
    except requests.exceptions.Timeout as e:
        print(f"Ollama server timeout: {e}")
        return False, f"Timeout error: {e}"
    except Exception as e:
        print(f"Unexpected error checking Ollama server: {e}")
        return False, f"Unexpected error: {e}"

def image_to_text(image):
    """Convert image to text using Ollama deepseek-ocr:latest model"""
    print("Starting image_to_text function...")
    
    # Check if Ollama server is running
    print("Checking Ollama server...")
    is_running, message = check_ollama_server()
    if not is_running:
        raise HTTPException(
            status_code=500, 
            detail=f"Ollama server issue: {message}. Please ensure Ollama is running."
        )
    
    print(f"Ollama server status: {message}")
    
    try:
        # Convert PIL Image to bytes
        print("Converting image to bytes...")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        print(f"Image bytes size: {len(img_bytes)}")
        
        # Use Ollama to extract text from image - SIMPLIFIED APPROACH
        print("Calling Ollama generate with deepseek-ocr:latest...")
        
        # Try the chat endpoint which might be more reliable
        response = ollama.chat(
            model="deepseek-ocr:latest",
            messages=[
                {
                    'role': 'user',
                    'content': 'Extract all text from this image. Return only the text.',
                    'images': [img_bytes]
                }
            ]
        )
        
        print(f"Ollama response received")
        
        if 'message' in response and 'content' in response['message']:
            extracted_text = response['message']['content']
            print(f"Extracted {len(extracted_text)} characters")
            return extracted_text
        else:
            print(f"Unexpected response format: {response}")
            raise HTTPException(
                status_code=500, 
                detail="Unexpected response format from Ollama"
            )
            
    except Exception as e:
        print(f"Error in image_to_text: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error extracting text: {str(e)}"
        )

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    print(f"\n=== NEW UPLOAD REQUEST ===")
    print(f"File: {file.filename}, Type: {file.content_type}")
    
    try:
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        # Convert bytes to image
        image = bytes_to_image(contents)
        print(f"Image created: {image.size}, mode: {image.mode}")

        # Extracting the text from image
        print("Starting OCR extraction...")
        extracted_text = image_to_text(image)

        print(f"OCR Complete!")
        print(f"Extracted text length: {len(extracted_text)}")
        print(f"First 500 chars: {extracted_text[:500]}")
        
        return {
            "content_type": file.content_type,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text)
        }
        
    except HTTPException as http_err:
        print(f"HTTP Exception: {http_err.detail}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# ========== TEST ENDPOINTS ==========

@app.get("/test-ollama-basic")
async def test_ollama_basic():
    """Test basic Ollama communication without images"""
    try:
        print("Testing basic Ollama communication...")
        response = ollama.generate(
            model="deepseek-ocr:latest",
            prompt="Hello, are you working? Just say 'YES' if you're working.",
            options={'temperature': 0.1}
        )
        return {
            "status": "success",
            "response": response.get('response', ''),
            "model": response.get('model', ''),
            "full_response": response
        }
    except Exception as e:
        print(f"Basic test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@app.get("/test-ollama-image")
async def test_ollama_image():
    """Test Ollama with a simple generated image (text image)"""
    try:
        print("Testing Ollama with generated image...")
        
        # Create a simple image with text
        from PIL import ImageDraw, ImageFont
        
        # Create a blank image
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        
        # Try to use a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add text to image
        d.text((10, 10), "This is a test image.", fill='black', font=font)
        d.text((10, 50), "Testing OCR with Ollama.", fill='black', font=font)
        d.text((10, 90), "If you can read this, it works!", fill='black', font=font)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        print(f"Created test image: {img.size}, bytes: {len(img_bytes)}")
        
        # Send to Ollama
        response = ollama.chat(
            model="deepseek-ocr:latest",
            messages=[
                {
                    'role': 'user',
                    'content': 'Extract all text from this image.',
                    'images': [img_bytes]
                }
            ]
        )
        
        extracted_text = response.get('message', {}).get('content', '')
        
        return {
            "status": "success",
            "image_size": img.size,
            "extracted_text": extracted_text,
            "response_summary": f"Extracted {len(extracted_text)} characters"
        }
        
    except Exception as e:
        print(f"Image test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@app.post("/test-simple-upload")
async def test_simple_upload():
    """Test endpoint that simulates upload with a simple text image"""
    try:
        print("Creating test image for simple upload...")
        
        # Create a simple test image
        from PIL import ImageDraw, ImageFont
        
        img = Image.new('RGB', (500, 300), color='white')
        d = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        lines = [
            "TEST IMAGE FOR OCR",
            "==================",
            "This is line 1 of test text.",
            "This is line 2 with numbers: 12345",
            "This is line 3 with symbols: !@#$%",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        y = 20
        for line in lines:
            d.text((20, y), line, fill='black', font=font)
            y += 40
        
        # Process like upload-image endpoint
        print("Processing image...")
        extracted_text = image_to_text(img)
        
        return {
            "status": "success",
            "image_size": img.size,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text)
        }
        
    except Exception as e:
        print(f"Simple upload test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify Ollama is working"""
    is_running, message = check_ollama_server()
    
    # Try to get model info
    model_info = {}
    try:
        models = ollama.list()
        model_info = models
    except Exception as e:
        model_info = {"error": str(e)}
    
    return {
        "ollama_server": {
            "running": is_running,
            "message": message
        },
        "models": model_info,
        "status": "healthy" if is_running else "unhealthy"
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
    print("Starting FastAPI server...")
    print("Checking initial Ollama server status...")
    is_running, message = check_ollama_server()
    print(f"Ollama server: {message}")
    
    if is_running:
        try:
            models = ollama.list()
            print(f"Available models: {models.get('models', [])}")
        except Exception as e:
            print(f"Warning: Could not list models: {e}")
    
    print("\nTest endpoints available:")
    print("1. GET  /health - Health check")
    print("2. GET  /test-ollama-basic - Test Ollama without images")
    print("3. GET  /test-ollama-image - Test Ollama with generated image")
    print("4. POST /test-simple-upload - Test full pipeline")
    print("5. POST /upload-image/ - Main upload endpoint")
    print("6. POST /generate-image/ - Image generation endpoint")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
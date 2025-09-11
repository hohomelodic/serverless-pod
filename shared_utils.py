"""
Shared utilities for image processing and common functions
"""

import base64
import io
from PIL import Image
from typing import Union

def decode_base64_image(image_b64: str) -> Image.Image:
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        image_data = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_data = buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')

def validate_image_input(image_b64: str, image_name: str) -> bool:
    """Validate that image input is properly formatted"""
    if not image_b64:
        raise ValueError(f"Missing {image_name}")
    
    try:
        decode_base64_image(image_b64)
        return True
    except Exception as e:
        raise ValueError(f"Invalid {image_name}: {str(e)}")

def resize_image_to_fit(image: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
    """Resize image to fit within max_size while maintaining aspect ratio"""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

"""
RunPod Serverless Handler for Qwen Edit & Generate API
API endpoint for product placement in real rooms and generated environments
"""

import runpod
import json
import time
from typing import Dict, Any, Optional
import torch
from qwen_edit_service import QwenEditService
from qwen_gen_service import QwenGenService
from shared_utils import decode_base64_image, encode_image_to_base64, validate_image_input

# Global services - will be loaded lazily on first request
edit_service = None
gen_service = None

def load_services():
    """Load Qwen services lazily on first request"""
    global edit_service, gen_service
    
    if edit_service is None:
        print("Loading Qwen Edit service...")
        edit_service = QwenEditService()
        print("Edit service loaded successfully!")
    
    if gen_service is None:
        print("Loading Qwen Generate service...")
        gen_service = QwenGenService()
        print("Generate service loaded successfully!")
    
    return edit_service, gen_service

def process_edit_request(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process edit request - place product in real room
    
    Args:
        job_input: Dictionary containing:
            - product_image: Base64 encoded product image
            - room_image: Base64 encoded room image
            - instructions: Text instructions for placement
            - placement_coordinates: Optional specific coordinates (x, y)
    
    Returns:
        Dictionary with processed image and metadata
    """
    try:
        # Load services lazily
        edit_svc, _ = load_services()
        
        # Validate required inputs
        validate_image_input(job_input.get("product_image"), "product_image")
        validate_image_input(job_input.get("room_image"), "room_image")
        
        # Extract parameters
        product_image_b64 = job_input.get("product_image")
        room_image_b64 = job_input.get("room_image")
        instructions = job_input.get("instructions", "")
        placement_coordinates = job_input.get("placement_coordinates")
        # Note: Edit API uses original room image size, no custom output_size
        
        # Decode images
        product_image = decode_base64_image(product_image_b64)
        room_image = decode_base64_image(room_image_b64)
        
        # Process with Edit service
        result = edit_svc.place_product_in_room(
            room_image=room_image,
            product_image=product_image,
            instructions=instructions,
            placement_coordinates=placement_coordinates
        )
        
        if result.get("error"):
            return result
        
        # Encode result image
        result_image_b64 = encode_image_to_base64(result["processed_image"])
        
        return {
            "success": True,
            "action": "edit",
            "processed_image": result_image_b64,
            "placement_info": result.get("placement_info", {}),
            "processing_time": result.get("processing_time", 0)
        }
        
    except Exception as e:
        return {
            "error": f"Edit processing failed: {str(e)}",
            "success": False
        }

def process_generate_request(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process generate request - create environment and place product
    
    Args:
        job_input: Dictionary containing:
            - product_image: Base64 encoded product image
            - instructions: Text instructions for environment and placement
            - output_size: Optional custom output dimensions
    
    Returns:
        Dictionary with generated image and metadata
    """
    try:
        # Load services lazily
        _, gen_svc = load_services()
        
        # Validate required inputs
        validate_image_input(job_input.get("product_image"), "product_image")
        
        # Extract parameters
        product_image_b64 = job_input.get("product_image")
        instructions = job_input.get("instructions", "")
        output_size = job_input.get("output_size")
        
        # Decode image
        product_image = decode_base64_image(product_image_b64)
        
        # Process with Generate service
        result = gen_svc.generate_product_environment(
            product_image=product_image,
            instructions=instructions,
            output_size=output_size
        )
        
        if result.get("error"):
            return result
        
        # Encode result image
        result_image_b64 = encode_image_to_base64(result["processed_image"])
        
        return {
            "success": True,
            "action": "generate",
            "processed_image": result_image_b64,
            "generation_info": result.get("generation_info", {}),
            "processing_time": result.get("processing_time", 0)
        }
        
    except Exception as e:
        return {
            "error": f"Generate processing failed: {str(e)}",
            "success": False
        }

def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "qwen-edit-generate-api",
        "version": "2.0.0",
        "actions": ["edit", "generate", "health_check"],
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

# RunPod handler
def handler(event):
    """
    Main RunPod handler function
    """
    print(f"Worker Start")
    
    try:
        # Extract input data
        input_data = event['input']
        print(f"Received input: {list(input_data.keys())}")
        
        # Check if it's a health check
        if input_data.get("action") == "health_check":
            return health_check()
        
        # Route to appropriate service based on action
        action = input_data.get("action", "edit")
        
        if action == "edit":
            result = process_edit_request(input_data)
        elif action == "generate":
            result = process_generate_request(input_data)
        else:
            return {
                "error": f"Invalid action: {action}. Supported actions: edit, generate, health_check",
                "success": False
            }
        
        print(f"Processing completed successfully")
        return result
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

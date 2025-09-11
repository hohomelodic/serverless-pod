# RunPod Serverless Deployment Guide

This guide will walk you through deploying your smart dual-API Qwen Edit & Generate service to RunPod Serverless.

## Prerequisites

- RunPod account (sign up at [runpod.io](https://runpod.io))
- Python 3.x installed locally
- Docker installed and configured
- Command line access

## Step 1: Local Setup

### 1.1 Create Virtual Environment
```bash
# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 1.2 Install Dependencies
```bash
# Install RunPod SDK
pip install runpod

# Install other dependencies
pip install -r requirements.txt
```

## Step 2: Test Locally

### 2.1 Test the Handler
```bash
python rp_handler.py
```

You should see output similar to:
```
--- Starting Serverless Worker |  Version 1.7.13 ---
INFO   | Using test_deployment.json as job input.
DEBUG  | Retrieved local job: {'input': {'action': 'health_check'}, 'id': 'local_test'}
INFO   | local_test | Started.
Worker Start
Received input: ['action']
Processing completed successfully
INFO   | Job local_test completed successfully.
```

### 2.2 Test Both APIs
The service supports two main actions:

**Edit API Test:**
```json
{
    "input": {
        "action": "edit",
        "product_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "room_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
        "instructions": "Place the product on the floor near the window"
    }
}
```

**Generate API Test:**
```json
{
    "input": {
        "action": "generate",
        "product_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "instructions": "Create a modern living room and place the product near a window",
        "environment_type": "living_room"
    }
}
```

## Step 3: Build and Push Docker Image

### 3.1 Build Docker Image
Replace `[YOUR_USERNAME]` with your Docker Hub username:

```bash
docker build --platform linux/amd64 --tag [YOUR_USERNAME]/qwen-dual-api-serverless .
```

### 3.2 Push to Docker Hub
```bash
docker push [YOUR_USERNAME]/qwen-dual-api-serverless:latest
```

## Step 4: Deploy to RunPod

### 4.1 Create Serverless Endpoint
1. Go to [RunPod Console](https://console.runpod.io)
2. Navigate to **Serverless** section
3. Click **New Endpoint**
4. Click **Import from Docker Registry**

### 4.2 Configure Endpoint
- **Container Image**: `docker.io/[YOUR_USERNAME]/qwen-dual-api-serverless:latest`
- **Endpoint Type**: Queue
- **GPU Configuration**: 
  - Check **16 GB GPUs** (recommended for Qwen models)
  - Or **24 GB GPUs** for better performance
- **Timeout**: 300 seconds (5 minutes)
- **Max Workers**: 1-3 (depending on your needs)

### 4.3 Deploy
Click **Deploy Endpoint** and wait for deployment to complete.

## Step 5: Test Your Endpoint

### 5.1 Test in RunPod Console
1. Go to your endpoint's detail page
2. Click the **Requests** tab
3. Use this test input:
```json
{
    "input": {
        "action": "health_check"
    }
}
```
4. Click **Run**

### 5.2 Test with cURL
```bash
curl -X POST https://your-endpoint-id-0-0-0-0.runpod.net \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "health_check"
    }
  }'
```

### 5.3 Test Edit API
```bash
curl -X POST https://your-endpoint-id-0-0-0-0.runpod.net \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "edit",
      "product_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "room_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
      "instructions": "Place the product on the floor near the window"
    }
  }'
```

### 5.4 Test Generate API
```bash
curl -X POST https://your-endpoint-id-0-0-0-0.runpod.net \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "generate",
      "product_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
      "instructions": "Create a modern living room and place the product near a window",
      "environment_type": "living_room"
    }
  }'
```

## Step 6: Integration with Shopify

### 6.1 Frontend Integration
Integrate both APIs into your Shopify app:

```javascript
// Edit API - Product in Real Room
const editResponse = await fetch('https://your-endpoint-id-0-0-0-0.runpod.net', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    input: {
      action: "edit",
      product_image: productImageBase64,
      room_image: roomImageBase64,
      instructions: userInstructions
    }
  })
});

// Generate API - Product in Generated Environment
const generateResponse = await fetch('https://your-endpoint-id-0-0-0-0.runpod.net', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    input: {
      action: "generate",
      product_image: productImageBase64,
      instructions: environmentInstructions,
      environment_type: "living_room"
    }
  })
});
```

### 6.2 Shopify App Structure
```
shopify-app/
├── web/
│   ├── frontend/
│   │   └── pages/
│   │       └── products/
│   │           └── [id].tsx    # Product page with both "Try in Room" and "Generate Environment" buttons
│   └── api/
│       └── qwen-dual-api/
│           └── route.ts        # API route to call RunPod endpoint
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure you have sufficient GPU memory (16GB+ recommended)
   - Check if the Qwen models are accessible
   - Verify all dependencies are installed correctly

2. **Docker Build Issues**
   - Make sure you're building for the correct platform: `--platform linux/amd64`
   - Check that all files are copied correctly in the Dockerfile

3. **Endpoint Timeout**
   - Increase the timeout setting in RunPod console
   - Optimize your model loading time
   - Consider using smaller models for faster startup

4. **Memory Issues**
   - Use a GPU with more VRAM (24GB+ recommended)
   - Optimize image sizes before sending to the API
   - Consider using model quantization

### Debug Mode
Add debug logging by modifying the handler:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Cost Optimization

1. **GPU Selection**: Choose the right GPU tier based on your needs
2. **Worker Scaling**: Set appropriate min/max workers
3. **Timeout Settings**: Optimize timeout to avoid unnecessary costs
4. **Model Optimization**: Use quantized models for faster inference

## Monitoring

Monitor your endpoint:
- **Usage**: Track API calls and processing time
- **Costs**: Monitor GPU usage and associated costs
- **Performance**: Track response times and error rates
- **Logs**: Check RunPod console for detailed logs

## Next Steps

1. **Production Optimization**: Implement caching, rate limiting, and error handling
2. **Advanced Features**: Add support for batch processing, multiple products, etc.
3. **Integration**: Build a complete Shopify app with user management
4. **Scaling**: Implement auto-scaling based on demand

---

**Note**: This is a basic deployment guide. For production use, consider additional security, monitoring, and optimization measures.

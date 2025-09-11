# Qwen Edit & Generate - RunPod Serverless API

A smart dual-API RunPod serverless service for AI-powered product placement using Qwen models and PyTorch. Supports both editing products into real rooms and generating new environments with products.

## 🚀 Features

### Edit API - Product in Real Room
- **AI-Powered Placement**: Uses Qwen models to intelligently place products in real room images
- **Room Analysis**: Automatically detects floors, walls, and furniture for optimal placement
- **Custom Instructions**: Accepts natural language instructions for placement preferences
- **Realistic Rendering**: Adds shadows and proper scaling for realistic product placement

### Generate API - Product in Generated Environment
- **Environment Generation**: Creates new room environments based on instructions
- **Smart Placement**: Places products in generated environments with proper context
- **Multiple Room Types**: Supports living room, bedroom, office, and custom environments
- **Creative Control**: Full control over environment style and product placement

### Smart Architecture
- **Single Endpoint**: Cost-efficient RunPod deployment with action-based routing
- **Shared Utilities**: Common image processing and validation functions
- **Production Ready**: Clean, organized, and scalable codebase
- **RunPod Optimized**: Built specifically for RunPod serverless deployment

## 📁 Project Structure

```
├── rp_handler.py              # Main RunPod handler (routes to edit/gen)
├── qwen_edit_service.py       # Edit API: Product in real room
├── qwen_gen_service.py        # Generate API: Product in generated environment
├── shared_utils.py            # Common utilities (image processing, validation)
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration for RunPod
├── test_deployment.json      # Health check test file
├── .gitignore               # Git ignore rules
├── DEPLOYMENT.md            # Step-by-step deployment guide
└── README.md                # This file
```

## 🛠️ Quick Start

### 1. Local Testing
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install runpod
pip install -r requirements.txt

# Test locally
python rp_handler.py
```

### 2. Deploy to RunPod
Follow the detailed deployment guide in [DEPLOYMENT.md](DEPLOYMENT.md):

1. **Build Docker Image**:
   ```bash
   docker build --platform linux/amd64 --tag [YOUR_USERNAME]/qwen-dual-api-serverless .
   docker push [YOUR_USERNAME]/qwen-dual-api-serverless:latest
   ```

2. **Deploy Endpoint**:
   - Go to [RunPod Console](https://console.runpod.io) → Serverless
   - Click "New Endpoint" → "Import from Docker Registry"
   - Container Image: `docker.io/[YOUR_USERNAME]/qwen-dual-api-serverless:latest`
   - GPU: 16GB+ recommended
   - Timeout: 300 seconds

3. **Test Endpoint**:
   ```bash
   curl -X POST https://your-endpoint-id-0-0-0-0.runpod.net \
     -H "Content-Type: application/json" \
     -d '{"input": {"action": "health_check"}}'
   ```

## 📡 API Usage

### Endpoint Structure
```
POST https://your-endpoint-id-0-0-0-0.runpod.net
```

### 1. Edit API - Product in Real Room

**Request Format:**
```json
{
  "input": {
    "action": "edit",
    "product_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "room_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "instructions": "Place the product on the floor near the window",
    "placement_coordinates": [400, 300]
  }
}
```

**Response Format:**
```json
{
  "success": true,
  "action": "edit",
  "processed_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "placement_info": {
    "instructions": "Place the product on the floor near the window",
    "coordinates": [400, 300],
    "processing_time": 2.45
  },
  "processing_time": 2.45
}
```

### 2. Generate API - Product in Generated Environment

**Request Format:**
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

**Response Format:**
```json
{
  "success": true,
  "action": "generate",
  "processed_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "generation_info": {
    "instructions": "Create a modern living room and place the product near a window",
    "environment_type": "living_room",
    "processing_time": 3.12
  },
  "processing_time": 3.12
}
```

### 3. Health Check

**Request Format:**
```json
{
  "input": {
    "action": "health_check"
  }
}
```

**Response Format:**
```json
{
  "status": "healthy",
  "service": "qwen-edit-generate-api",
  "version": "2.0.0",
  "actions": ["edit", "generate", "health_check"],
  "gpu_available": true,
  "gpu_count": 1
}
```

### Parameters

#### Edit API Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | Must be "edit" |
| `product_image` | string | Yes | Base64 encoded product image |
| `room_image` | string | Yes | Base64 encoded room image |
| `instructions` | string | No | Natural language placement instructions |
| `placement_coordinates` | array | No | Specific [x, y] coordinates for placement |

#### Generate API Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | Must be "generate" |
| `product_image` | string | Yes | Base64 encoded product image |
| `instructions` | string | No | Environment and placement instructions |
| `environment_type` | string | No | Type of environment (living_room, bedroom, office) |

## 🔧 Technical Details

### Models Used
- **Qwen2-VL-2B-Instruct**: For understanding room layouts, placement instructions, and environment generation
- **PyTorch**: For image processing and model inference
- **Transformers**: For loading and running Qwen models

### Architecture
- **Smart Routing**: Single endpoint with action-based request handling
- **Service Separation**: Dedicated services for edit and generate operations
- **Shared Utilities**: Common image processing, validation, and encoding functions
- **Error Handling**: Comprehensive error handling and validation

### Image Processing Pipeline

#### Edit API Pipeline
1. **Input Validation**: Validates product and room images
2. **Room Analysis**: Detects floors, walls, furniture, and lighting
3. **Placement Zone Identification**: Finds suitable areas for product placement
4. **Instruction Processing**: Interprets natural language placement preferences
5. **Product Scaling**: Resizes product to appropriate scale for room context
6. **Realistic Rendering**: Adds shadows and proper compositing
7. **Result Generation**: Returns processed image with placement metadata

#### Generate API Pipeline
1. **Input Validation**: Validates product image and instructions
2. **Environment Generation**: Creates new room environment based on type and instructions
3. **Product Placement**: Places product in generated environment with proper context
4. **Style Matching**: Ensures product matches environment style
5. **Result Generation**: Returns generated image with environment metadata

### Performance
- **Processing Time**: 2-5 seconds per request (depending on image size and GPU)
- **Memory Usage**: ~8-12GB VRAM for model loading
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Scalability**: Single endpoint design allows efficient resource utilization

## 🎯 Integration Examples

### Shopify Integration

To integrate with Shopify:

1. **Create Shopify App**: Use the Shopify CLI to create a new app
2. **Add Product Buttons**: Add "Try in Room" and "Generate Environment" buttons to product pages
3. **Frontend Integration**: Build your own interface or use existing templates
4. **API Calls**: Make requests to your RunPod endpoint from your Shopify app

Example Shopify app structure:
```
shopify-app/
├── web/
│   ├── frontend/
│   │   └── pages/
│   │       └── products/
│   │           └── [id].tsx    # Product page with both buttons
│   └── api/
│       └── qwen-dual-api/
│           └── route.ts        # API route to call RunPod endpoint
```

### JavaScript Integration Example

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
      instructions: "Place the product on the floor near the window"
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
      instructions: "Create a modern living room and place the product near a window",
      environment_type: "living_room"
    }
  })
});
```

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure sufficient GPU memory (16GB+ recommended)
   - Check if Qwen models are accessible from RunPod

2. **Image Processing Failures**:
   - Verify images are properly base64 encoded
   - Check image formats (JPEG, PNG supported)
   - Ensure images are not too large (>10MB)

3. **Timeout Errors**:
   - Increase RunPod endpoint timeout to 300+ seconds
   - Optimize image sizes before sending

4. **Memory Issues**:
   - Use smaller batch sizes
   - Consider using CPU fallback for smaller images

### Debug Mode

Add debug logging by setting environment variable:
```bash
export DEBUG=1
```

## 📊 Monitoring

Monitor your RunPod endpoint:
- **Usage**: Track API calls and processing time
- **Errors**: Monitor failed requests and error rates
- **Costs**: Keep track of GPU usage and associated costs
- **Performance**: Monitor response times and throughput

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in this repository
- Check RunPod documentation: [docs.runpod.io](https://docs.runpod.io)
- Join RunPod Discord community

---

**Note**: This is a simplified implementation. For production use, consider adding:
- Input validation and sanitization
- Rate limiting and authentication
- Error handling and retry logic
- Caching for frequently used models
- Monitoring and logging

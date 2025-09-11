# Qwen Edit - RunPod Serverless API

A RunPod serverless API for AI-powered product placement in room images using Qwen models and PyTorch.

## 🚀 Features

- **AI-Powered Placement**: Uses Qwen models to intelligently place products in room images
- **Room Analysis**: Automatically detects floors, walls, and furniture for optimal placement
- **Custom Instructions**: Accepts natural language instructions for placement preferences
- **Realistic Rendering**: Adds shadows and proper scaling for realistic product placement
- **RunPod Optimized**: Built specifically for RunPod serverless deployment

## 📁 Project Structure

```
├── rp_handler.py          # Main RunPod serverless handler
├── qwen_edit_service.py   # Core Qwen Edit service with PyTorch
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration for RunPod
├── test_input.json       # Test input for local testing
├── DEPLOYMENT.md         # Step-by-step deployment guide
└── README.md             # This file
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
   docker build --platform linux/amd64 --tag [YOUR_USERNAME]/qwen-edit-serverless .
   docker push [YOUR_USERNAME]/qwen-edit-serverless:latest
   ```

2. **Deploy Endpoint**:
   - Go to [RunPod Console](https://console.runpod.io) → Serverless
   - Click "New Endpoint" → "Import from Docker Registry"
   - Container Image: `docker.io/[YOUR_USERNAME]/qwen-edit-serverless:latest`
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

### Request Format
```json
{
  "input": {
    "room_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "product_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "instructions": "Place the product on the floor near the window",
    "placement_coordinates": [400, 300]
  }
}
```

### Response Format
```json
{
  "success": true,
  "result": {
    "success": true,
    "processed_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "placement_info": {
      "location": [400, 300],
      "type": "floor",
      "product_size": [120, 150],
      "room_analysis": {...}
    },
    "processing_time": 2.45
  }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `room_image` | string | Yes | Base64 encoded room image |
| `product_image` | string | Yes | Base64 encoded product image |
| `instructions` | string | No | Natural language placement instructions |
| `placement_coordinates` | array | No | Specific [x, y] coordinates for placement |

## 🔧 Technical Details

### Models Used
- **Qwen-VL-Chat**: For understanding room layouts and placement instructions
- **PyTorch**: For image processing and model inference
- **OpenCV**: For computer vision tasks (edge detection, contour analysis)

### Image Processing Pipeline
1. **Room Analysis**: Detects floors, walls, furniture, and lighting
2. **Placement Zone Identification**: Finds suitable areas for product placement
3. **Instruction Processing**: Interprets natural language placement preferences
4. **Product Scaling**: Resizes product to appropriate scale for room context
5. **Realistic Rendering**: Adds shadows and proper compositing
6. **Result Generation**: Returns processed image with placement metadata

### Performance
- **Processing Time**: 2-5 seconds per request (depending on image size and GPU)
- **Memory Usage**: ~8-12GB VRAM for model loading
- **Concurrent Requests**: Supports multiple simultaneous requests

## 🎯 Shopify Integration

To integrate with Shopify:

1. **Create Shopify App**: Use the Shopify CLI to create a new app
2. **Add Product Button**: Add a "Try in Room" button to product pages
3. **Frontend Integration**: Use the provided HTML interface or build your own
4. **API Calls**: Make requests to your RunPod endpoint from your Shopify app

Example Shopify app structure:
```
shopify-app/
├── web/
│   ├── frontend/
│   │   └── pages/
│   │       └── products/
│   │           └── [id].tsx    # Product page with "Try in Room" button
│   └── api/
│       └── qwen-edit/
│           └── route.ts        # API route to call RunPod endpoint
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

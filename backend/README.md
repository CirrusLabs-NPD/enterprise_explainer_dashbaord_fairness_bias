# ML Explainer Dashboard Backend

Enterprise-grade ML model explainability and monitoring system backend built with FastAPI.

## Features

- **Model Management**: Upload, manage, and deploy ONNX models
- **Advanced Explainability**: SHAP, LIME, and custom explanation algorithms
- **Real-time Monitoring**: Drift detection, performance monitoring, and alerting
- **Data Processing**: Upload, preprocess, and quality assessment of datasets
- **WebSocket Support**: Real-time updates and notifications
- **Scalable Architecture**: Multiprocessing support for compute-intensive tasks

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL (optional, SQLite for development)
- Redis (optional)

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start development server**:
   ```bash
   python run_dev.py
   ```

3. **Access the API**:
   - Dashboard: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - WebSocket Test: http://localhost:8000/api/v1/ws/test-page

## API Endpoints

### Models
- `POST /api/v1/models/upload` - Upload a new model
- `GET /api/v1/models/` - List all models
- `GET /api/v1/models/{model_id}` - Get model details
- `POST /api/v1/models/{model_id}/predict` - Make predictions
- `POST /api/v1/models/{model_id}/explain` - Generate explanations

### Data
- `POST /api/v1/data/upload` - Upload dataset
- `GET /api/v1/data/` - List datasets
- `GET /api/v1/data/{dataset_id}/statistics` - Get dataset statistics
- `GET /api/v1/data/{dataset_id}/quality` - Assess data quality

### Monitoring
- `POST /api/v1/monitoring/configure/{model_id}` - Configure monitoring
- `POST /api/v1/monitoring/drift/detect/{model_id}` - Detect drift
- `GET /api/v1/monitoring/alerts/{model_id}` - Get alerts

### WebSocket
- `ws://localhost:8000/api/v1/ws/connect` - WebSocket connection
- Real-time updates for model predictions, explanations, and alerts

## Architecture

### Core Components

1. **FastAPI Application** (`app/main.py`)
   - ASGI application with lifespan management
   - CORS and security middleware
   - API routing and documentation

2. **Services** (`app/services/`)
   - `ModelService`: Model management and inference
   - `ExplanationService`: SHAP, LIME, and custom explanations
   - `MonitoringService`: Drift detection and alerting
   - `DataService`: Data processing and quality assessment

3. **Worker Pool** (`app/core/worker_pool.py`)
   - Multiprocessing support for CPU-intensive tasks
   - Task queuing and prioritization
   - Automatic retry and error handling

4. **Database** (`app/core/database.py`)
   - PostgreSQL/SQLite support
   - Connection pooling
   - Migration management

5. **WebSocket Manager** (`app/core/websocket_manager.py`)
   - Real-time communication
   - Model subscriptions
   - Broadcast capabilities

### Key Features

- **Multiprocessing Support**: CPU-intensive tasks (SHAP, LIME) run in separate processes
- **Async/Await**: Non-blocking I/O operations
- **Type Safety**: Full type hints with Pydantic models
- **Monitoring**: Prometheus metrics and structured logging
- **Security**: JWT authentication and rate limiting
- **Scalability**: Horizontal scaling support

## Configuration

Environment variables (see `app/config.py`):

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/db
DATABASE_MIN_CONNECTIONS=1
DATABASE_MAX_CONNECTIONS=20

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Worker Pool
MAX_WORKERS=4
MAX_CPU_WORKERS=2
MAX_IO_WORKERS=8

# Security
SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Monitoring
DRIFT_THRESHOLD=0.1
PERFORMANCE_THRESHOLD=0.05
MONITORING_INTERVAL=300
```

## Development

### Project Structure

```
backend/
├── app/
│   ├── api/v1/endpoints/     # API endpoints
│   ├── core/                 # Core utilities
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic
│   └── main.py              # FastAPI application
├── requirements.txt         # Python dependencies
├── run_dev.py              # Development server
└── README.md               # This file
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Code Quality

```bash
# Format code
black app/
isort app/

# Type checking
mypy app/

# Linting
flake8 app/
```

## Deployment

### Docker

```bash
# Build image
docker build -t ml-explainer-backend .

# Run container
docker run -p 8000:8000 ml-explainer-backend
```

### Production

1. **Set production environment variables**
2. **Use PostgreSQL database**
3. **Configure Redis for caching**
4. **Set up reverse proxy (nginx)**
5. **Configure monitoring and logging**

## API Documentation

- Interactive API docs: `/docs`
- OpenAPI schema: `/openapi.json`
- ReDoc documentation: `/redoc`

## WebSocket Protocol

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/connect?token=your_token');
```

### Messages
```javascript
// Subscribe to model updates
ws.send(JSON.stringify({
  type: 'subscribe_model',
  model_id: 'model_123'
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

## Support

For issues and questions:
- Check the API documentation at `/docs`
- Review the WebSocket test page at `/api/v1/ws/test-page`
- Monitor health at `/health`
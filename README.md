# ML Explainer Dashboard

A comprehensive, **enterprise-grade** Machine Learning Explainer Dashboard with advanced analytics, security, and monitoring capabilities. Built with React, TypeScript, FastAPI, and designed for production deployment with Docker and Kubernetes.

## 🏢 Enterprise Features

### 🔐 **Security & Authentication**
- **JWT-based Authentication** with role-based access control (RBAC)
- **Multi-role Support** - Admin, Analyst, Viewer permissions
- **Session Management** with automatic token refresh
- **Route Protection** with permission-based access guards
- **Rate Limiting** with visual indicators and monitoring

### 📊 **Advanced Analytics & ML**
- **Model Performance Monitoring** with real-time metrics
- **Data Drift Detection** with statistical analysis and alerts  
- **SHAP-based Explanations** - Waterfall charts, feature importance, dependence plots
- **Fairness Analysis** with bias detection and mitigation recommendations
- **Root Cause Analysis** for model performance degradation
- **A/B Testing Framework** for model comparison and validation

### 🏗️ **Enterprise Infrastructure**
- **Docker Containerization** with multi-stage builds
- **CI/CD Pipeline** with automated testing and deployment
- **Health Monitoring** with Prometheus and Grafana integration
- **Error Boundaries** with comprehensive error handling
- **Notification System** with real-time alerts and filtering
- **Data Connectors** for enterprise data sources (SQL, NoSQL, APIs)

### 📈 **Business Intelligence**
- **Executive Dashboard** with KPIs and business metrics
- **Compliance Reporting** for regulatory frameworks (GDPR, CCPA, etc.)
- **Custom Dashboard Builder** with drag-and-drop interface
- **Automated Reporting** with scheduled generation and distribution
- **Data Quality Monitoring** with validation rules and alerts

## 🛠️ Technology Stack

### **Frontend**
- **React 18** with TypeScript and modern hooks
- **Vite** for fast development and optimized builds
- **Tailwind CSS** with custom design system
- **Framer Motion** for smooth animations
- **Recharts & D3.js** for advanced data visualizations
- **React Testing Library & Vitest** for comprehensive testing

### **Backend**
- **FastAPI** with async support and automatic API documentation
- **PostgreSQL** for data persistence with Redis for caching
- **JWT Authentication** with bcrypt password hashing
- **Pydantic** for data validation and serialization
- **WebSocket** support for real-time updates
- **Prometheus** integration for metrics and monitoring

### **DevOps & Infrastructure**
- **Docker** with optimized multi-stage builds
- **GitHub Actions** for CI/CD automation
- **Environment-based Configuration** for development/staging/production
- **Health Checks** and graceful shutdown handling
- **Grafana Dashboards** for observability and monitoring

## 🏗️ Project Structure

```
explainer_dashboard/
├── backend/                    # FastAPI backend application
│   ├── app/
│   │   ├── api/v1/            # API endpoints and routing
│   │   ├── core/              # Authentication, security, database
│   │   ├── models/            # Data models and schemas
│   │   └── services/          # Business logic and enterprise services
│   ├── tests/                 # Backend test suite
│   └── main.py               # Application entry point
├── src/                       # React frontend application
│   ├── components/
│   │   ├── auth/             # Authentication components
│   │   ├── charts/           # Data visualization components
│   │   ├── common/           # Reusable UI components
│   │   └── layout/           # Layout and navigation
│   ├── contexts/             # React contexts (Auth, Theme)
│   ├── pages/                # Page components and features
│   ├── services/             # API services and data management
│   ├── utils/                # Utility functions and helpers
│   └── __tests__/            # Frontend test suites
├── .github/workflows/         # CI/CD pipeline configurations
├── docs/                     # Additional documentation
├── docker-compose.yml        # Multi-service orchestration
├── Dockerfile               # Container configuration
└── README.md               # This file
```

## 🚦 Quick Start

### **Prerequisites**
- **Node.js 18+** and npm/yarn
- **Python 3.9+** and pip
- **Docker & Docker Compose** (for containerized deployment)

### **Option 1: Docker Development (Recommended)**

1. **Clone and start all services**
   ```bash
   git clone <repository-url>
   cd explainer_dashboard
   
   # Start all services (frontend, backend, database, monitoring)
   docker-compose up --build
   ```

2. **Access the application**
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Grafana Dashboard**: http://localhost:3001

3. **Demo Accounts**
   - **Admin**: `admin` / `admin123`
   - **Analyst**: `analyst` / `analyst123`  
   - **Viewer**: `viewer` / `viewer123`

### **Option 2: Local Development**

1. **Frontend Setup**
   ```bash
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   ```

2. **Backend Setup**
   ```bash
   cd backend
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start backend server
   uvicorn app.main:app --reload --port 8000
   ```

3. **Environment Configuration**
   ```bash
   # Frontend (.env)
   VITE_API_URL=http://localhost:8000
   VITE_APP_NAME=ML Explainer Dashboard
   
   # Backend (.env)
   DATABASE_URL=postgresql://user:password@localhost/ml_dashboard
   SECRET_KEY=your-secret-key-here
   REDIS_URL=redis://localhost:6379
   ```

## 🧪 Testing

### **Run All Tests**
```bash
# Frontend tests
npm run test
npm run test:coverage

# Backend tests  
cd backend
pytest

# End-to-end tests
npm run test:e2e
```

### **Test Coverage**
- **Unit Tests**: Components, services, utilities
- **Integration Tests**: Authentication flows, API endpoints
- **E2E Tests**: Critical user journeys and workflows

## 🚀 Production Deployment

### **Docker Production Build**
```bash
# Build optimized production images
docker-compose -f docker-compose.prod.yml up --build

# Or build individual services
docker build -t ml-dashboard-frontend .
docker build -t ml-dashboard-backend ./backend
```

### **Environment Variables (Production)**
```bash
# Database
DATABASE_URL=postgresql://user:password@db:5432/ml_dashboard
REDIS_URL=redis://redis:6379

# Security
SECRET_KEY=your-production-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
```

## 🎯 Key Features & Capabilities

### **🔍 ML Model Explainability**
- **SHAP Integration**: Waterfall charts, force plots, summary plots
- **Feature Importance**: Global and local explanations
- **Partial Dependence**: Understanding feature relationships
- **Individual Predictions**: Detailed prediction breakdowns
- **Counterfactual Analysis**: What-if scenario exploration

### **📊 Advanced Analytics**
- **Model Performance**: Accuracy, precision, recall, F1-score tracking
- **Data Drift Detection**: Statistical tests and distribution monitoring
- **Fairness Metrics**: Bias detection across demographic groups
- **Business Impact**: ROI calculations and business metric correlation
- **Comparative Analysis**: A/B testing and model versioning

### **🏢 Enterprise Management**
- **User Management**: Role-based permissions and access control
- **System Monitoring**: Health checks, performance metrics, error tracking
- **Compliance Reporting**: Automated generation for regulatory requirements
- **Data Governance**: Lineage tracking and audit trails
- **Scalability**: Horizontal scaling with load balancing support

### **🔔 Real-time Features**
- **Live Notifications**: Model alerts, system status, drift detection
- **WebSocket Updates**: Real-time dashboard refreshes
- **Background Processing**: Asynchronous analysis and report generation
- **Rate Limiting**: API protection with visual feedback
- **Caching**: Redis-based performance optimization

## 📚 Documentation

- **[Architecture Guide](./ARCHITECTURE.md)** - Technical architecture and design decisions
- **[Deployment Guide](./DEPLOYMENT_GUIDE.md)** - Production deployment instructions
- **[API Reference](./backend/BACKEND_API_REFERENCE.md)** - Complete API documentation
- **[Implementation Summary](./IMPLEMENTATION_SUMMARY.md)** - Development roadmap and features

## 🔧 Configuration & Customization

### **Theme Customization**
```typescript
// src/contexts/ThemeContext.tsx
const customTheme = {
  colors: {
    primary: '#your-brand-color',
    secondary: '#your-secondary-color'
  }
}
```

### **Feature Toggles**
```javascript
// src/config/features.ts
export const FEATURES = {
  ADVANCED_ANALYTICS: true,
  A_B_TESTING: true,
  COMPLIANCE_REPORTING: true,
  REAL_TIME_MONITORING: true
}
```

## 🛡️ Security Considerations

- **Authentication**: JWT tokens with secure HTTP-only cookies
- **Authorization**: Role-based access control with permission matrices
- **Data Protection**: Input validation, SQL injection prevention
- **Rate Limiting**: API endpoint protection against abuse
- **CORS Configuration**: Controlled cross-origin resource sharing
- **Security Headers**: Comprehensive HTTP security headers

## 🔧 Troubleshooting

### **Common Issues**

1. **Port Conflicts**
   ```bash
   # Check running services
   docker ps
   lsof -i :3000  # Frontend port
   lsof -i :8000  # Backend port
   ```

2. **Database Connection**
   ```bash
   # Check database status
   docker-compose logs postgres
   ```

3. **Permission Issues**
   ```bash
   # Reset file permissions
   chmod +x scripts/setup.sh
   ```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### **Development Guidelines**
- Follow TypeScript best practices
- Write comprehensive tests
- Update documentation
- Follow conventional commit messages
- Ensure Docker builds pass

## 📄 License

MIT License - see [LICENSE](./LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check guides in `/docs` folder
- **Issues**: GitHub Issues for bug reports and features
- **Discussions**: GitHub Discussions for community support

---

**Built with ❤️ for Enterprise ML Operations**

🚀 **Ready for Production** | 🔒 **Enterprise Security** | 📊 **Advanced Analytics** | 🐳 **Containerized** | 🧪 **Fully Tested**
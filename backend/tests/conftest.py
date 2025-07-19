"""
Test configuration and fixtures
"""

import pytest
import asyncio
from httpx import AsyncClient
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient

from app.main import app
from app.core.mock_database import init_db, close_db
from app.core.auth import auth_handler


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db():
    """Setup test database"""
    await init_db()
    yield
    await close_db()


@pytest.fixture
def client() -> TestClient:
    """Test client for synchronous tests"""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_headers() -> dict:
    """Authentication headers for tests"""
    token_data = {"sub": "test_user", "user_id": "test_user_id", "role": "admin"}
    token = auth_handler.create_access_token(token_data)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers() -> dict:
    """Admin authentication headers"""
    token_data = {"sub": "admin", "user_id": "1", "role": "admin"}
    token = auth_handler.create_access_token(token_data)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def analyst_headers() -> dict:
    """Analyst authentication headers"""
    token_data = {"sub": "analyst", "user_id": "2", "role": "analyst"}
    token = auth_handler.create_access_token(token_data)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def viewer_headers() -> dict:
    """Viewer authentication headers"""
    token_data = {"sub": "viewer", "user_id": "3", "role": "viewer"}
    token = auth_handler.create_access_token(token_data)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_model_metadata() -> dict:
    """Sample model metadata for testing"""
    return {
        "name": "Test Model",
        "model_type": "classification",
        "feature_names": ["feature1", "feature2", "feature3"],
        "description": "A test model for unit tests",
        "version": "1.0.0",
        "framework": "sklearn"
    }


@pytest.fixture
def sample_prediction_data() -> dict:
    """Sample prediction data for testing"""
    return {
        "features": {
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0
        }
    }


@pytest.fixture
def sample_dataset() -> dict:
    """Sample dataset for testing"""
    return {
        "name": "Test Dataset",
        "description": "A test dataset",
        "data": [
            {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "target": 1},
            {"feature1": 2.0, "feature2": 3.0, "feature3": 4.0, "target": 0},
            {"feature1": 3.0, "feature2": 4.0, "feature3": 5.0, "target": 1}
        ]
    }


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test"""
    from app.core.monitoring import REGISTRY
    
    # Clear all metrics
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    
    yield
    
    # Clean up after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass


@pytest.fixture
def mock_model_file(tmp_path):
    """Create a mock model file for testing"""
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create a simple model
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save to file
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return model_path


@pytest.fixture
def mock_csv_file(tmp_path):
    """Create a mock CSV file for testing"""
    import pandas as pd
    
    # Create sample data
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 3, 4, 5, 6],
        "feature3": [3, 4, 5, 6, 7],
        "target": [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path
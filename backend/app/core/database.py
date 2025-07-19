"""
Database utilities and connection management
"""

import asyncio
import asyncpg
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class Database:
    """Database connection manager"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize database connection pool"""
        if self.initialized:
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                dsn=settings.DATABASE_URL,
                min_size=settings.DATABASE_MIN_CONNECTIONS,
                max_size=settings.DATABASE_MAX_CONNECTIONS,
                command_timeout=settings.DATABASE_TIMEOUT,
                server_settings={
                    'jit': 'off'  # Disable JIT for better performance on small queries
                }
            )
            
            # Test the connection
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            self.initialized = True
            logger.info("Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.initialized = False
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get database connection from pool"""
        if not self.initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch single row"""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch single value"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)


# Global database instance
database = Database()


# Database utility functions
@asynccontextmanager
async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get database connection"""
    async with database.get_connection() as conn:
        yield conn


async def init_db():
    """Initialize database and create tables"""
    await database.initialize()
    await create_tables()


async def close_db():
    """Close database connection"""
    await database.close()


async def create_tables():
    """Create database tables"""
    try:
        async with get_db() as db:
            # Create extensions
            await db.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
            await db.execute("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\"")
            
            # Model metadata table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    model_type VARCHAR NOT NULL,
                    framework VARCHAR NOT NULL DEFAULT 'sklearn',
                    file_path VARCHAR NOT NULL,
                    version VARCHAR NOT NULL DEFAULT '1.0.0',
                    description TEXT DEFAULT '',
                    
                    feature_names JSONB DEFAULT '[]',
                    target_names JSONB DEFAULT '[]',
                    feature_types JSONB DEFAULT '{}',
                    
                    preprocessing_steps JSONB DEFAULT '[]',
                    scaler_path VARCHAR,
                    encoder_path VARCHAR,
                    
                    hyperparameters JSONB DEFAULT '{}',
                    
                    training_metrics JSONB DEFAULT '{}',
                    validation_metrics JSONB DEFAULT '{}',
                    test_metrics JSONB DEFAULT '{}',
                    
                    model_size_bytes INTEGER,
                    training_time_seconds FLOAT,
                    inference_time_ms FLOAT,
                    
                    drift_threshold FLOAT DEFAULT 0.1,
                    performance_threshold FLOAT DEFAULT 0.05,
                    last_evaluation TIMESTAMP,
                    
                    status VARCHAR DEFAULT 'active',
                    created_by VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    tags JSONB DEFAULT '[]',
                    labels JSONB DEFAULT '{}'
                )
            """)
            
            # Dataset metadata table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    dataset_id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    description TEXT DEFAULT '',
                    file_path VARCHAR NOT NULL,
                    format VARCHAR NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    num_rows INTEGER NOT NULL,
                    num_columns INTEGER NOT NULL,
                    
                    column_names JSONB DEFAULT '[]',
                    column_types JSONB DEFAULT '{}',
                    statistics JSONB DEFAULT '{}',
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags JSONB DEFAULT '[]'
                )
            """)
            
            # Explanations table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS explanations (
                    explanation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    model_id VARCHAR NOT NULL,
                    method VARCHAR NOT NULL,
                    feature_names JSONB NOT NULL,
                    explanation JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms FLOAT,
                    
                    FOREIGN KEY (model_id) REFERENCES model_metadata(model_id) ON DELETE CASCADE
                )
            """)
            
            # Model monitoring configs table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_monitoring_configs (
                    model_id VARCHAR PRIMARY KEY,
                    monitoring_enabled BOOLEAN DEFAULT TRUE,
                    drift_detection_enabled BOOLEAN DEFAULT TRUE,
                    performance_monitoring_enabled BOOLEAN DEFAULT TRUE,
                    data_quality_monitoring_enabled BOOLEAN DEFAULT TRUE,
                    
                    drift_threshold FLOAT DEFAULT 0.1,
                    performance_threshold FLOAT DEFAULT 0.05,
                    data_quality_threshold FLOAT DEFAULT 0.8,
                    
                    drift_check_interval_hours INTEGER DEFAULT 24,
                    performance_check_interval_hours INTEGER DEFAULT 6,
                    data_quality_check_interval_hours INTEGER DEFAULT 12,
                    
                    alert_on_drift BOOLEAN DEFAULT TRUE,
                    alert_on_performance_drop BOOLEAN DEFAULT TRUE,
                    alert_on_data_quality_issues BOOLEAN DEFAULT TRUE,
                    alert_channels JSONB DEFAULT '[]',
                    
                    keep_monitoring_data_days INTEGER DEFAULT 30,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (model_id) REFERENCES model_metadata(model_id) ON DELETE CASCADE
                )
            """)
            
            # Drift reports table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS drift_reports (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    model_id VARCHAR NOT NULL,
                    report_type VARCHAR NOT NULL, -- 'data_drift' or 'model_drift'
                    
                    feature_drifts JSONB DEFAULT '{}',
                    overall_drift_score FLOAT,
                    drift_threshold FLOAT,
                    drift_detected BOOLEAN,
                    
                    reference_dataset_size INTEGER,
                    current_dataset_size INTEGER,
                    detection_method VARCHAR,
                    
                    performance_drift FLOAT,
                    current_metrics JSONB DEFAULT '{}',
                    reference_metrics JSONB DEFAULT '{}',
                    
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (model_id) REFERENCES model_metadata(model_id) ON DELETE CASCADE
                )
            """)
            
            # Data quality reports table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_reports (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    dataset_id VARCHAR NOT NULL,
                    total_rows INTEGER NOT NULL,
                    total_features INTEGER NOT NULL,
                    missing_values JSONB DEFAULT '{}',
                    duplicate_rows INTEGER DEFAULT 0,
                    outliers JSONB DEFAULT '{}',
                    data_types JSONB DEFAULT '{}',
                    quality_score FLOAT,
                    issues JSONB DEFAULT '[]',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    config_id VARCHAR,
                    model_id VARCHAR NOT NULL,
                    alert_type VARCHAR NOT NULL,
                    severity VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    message TEXT NOT NULL,
                    data JSONB DEFAULT '{}',
                    
                    status VARCHAR DEFAULT 'active',
                    acknowledged_by VARCHAR,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model performance metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_metrics (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    model_id VARCHAR NOT NULL,
                    metrics JSONB NOT NULL,
                    dataset_size INTEGER,
                    prediction_time_ms FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (model_id) REFERENCES model_metadata(model_id) ON DELETE CASCADE
                )
            """)
            
            # Predictions table (for tracking)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    model_id VARCHAR NOT NULL,
                    input_data JSONB NOT NULL,
                    predictions JSONB NOT NULL,
                    probabilities JSONB,
                    prediction_time_ms FLOAT,
                    model_version VARCHAR,
                    user_id VARCHAR,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (model_id) REFERENCES model_metadata(model_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_model_metadata_created_at ON model_metadata(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_model_metadata_status ON model_metadata(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_dataset_metadata_created_at ON dataset_metadata(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_explanations_model_id ON explanations(model_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_explanations_created_at ON explanations(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_drift_reports_model_id ON drift_reports(model_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_drift_reports_timestamp ON drift_reports(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_model_id ON alerts(model_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
            
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


async def reset_database():
    """Reset database (drop and recreate tables)"""
    try:
        async with get_db() as db:
            # Drop tables in reverse order
            tables = [
                "predictions",
                "model_performance_metrics",
                "alerts",
                "data_quality_reports",
                "drift_reports",
                "model_monitoring_configs",
                "explanations",
                "dataset_metadata",
                "model_metadata"
            ]
            
            for table in tables:
                await db.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            
            logger.info("Database tables dropped")
            
        # Recreate tables
        await create_tables()
        
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise


# Database migration utilities
async def get_database_version() -> str:
    """Get current database version"""
    try:
        async with get_db() as db:
            # Check if version table exists
            exists = await db.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'schema_version'
                )
            """)
            
            if not exists:
                # Create version table
                await db.execute("""
                    CREATE TABLE schema_version (
                        version VARCHAR PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert initial version
                await db.execute("""
                    INSERT INTO schema_version (version) VALUES ('1.0.0')
                """)
                
                return "1.0.0"
            
            # Get current version
            version = await db.fetchval("""
                SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1
            """)
            
            return version or "1.0.0"
            
    except Exception as e:
        logger.error(f"Error getting database version: {e}")
        return "unknown"


async def apply_migration(version: str, migration_sql: str):
    """Apply database migration"""
    try:
        async with get_db() as db:
            # Start transaction
            async with db.transaction():
                # Apply migration
                await db.execute(migration_sql)
                
                # Update version
                await db.execute("""
                    INSERT INTO schema_version (version) VALUES ($1)
                """, version)
                
                logger.info(f"Applied database migration to version {version}")
                
    except Exception as e:
        logger.error(f"Error applying migration {version}: {e}")
        raise


# Health check
async def check_database_health() -> dict:
    """Check database health"""
    try:
        async with get_db() as db:
            # Test connection
            result = await db.fetchval("SELECT 1")
            
            # Get database info
            db_info = await db.fetchrow("""
                SELECT 
                    version() as version,
                    current_database() as database,
                    current_user as user,
                    pg_database_size(current_database()) as size_bytes
            """)
            
            return {
                "status": "healthy",
                "version": db_info["version"],
                "database": db_info["database"],
                "user": db_info["user"],
                "size_bytes": db_info["size_bytes"],
                "pool_size": database.pool.get_size() if database.pool else 0,
                "pool_idle": database.pool.get_idle_size() if database.pool else 0
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
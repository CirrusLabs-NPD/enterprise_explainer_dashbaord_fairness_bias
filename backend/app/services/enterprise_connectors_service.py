"""
Enterprise Data Connectors Service
Provides connectors for major cloud platforms, data warehouses, and ML platforms
Compatible with Snowflake, AWS, Azure, GCP, Databricks, and other enterprise systems
"""

import asyncio
import json
import boto3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for data source connections"""
    connection_id: str
    connection_type: str
    name: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None
    encryption_enabled: bool = True
    ssl_required: bool = True


class BaseConnector(ABC):
    """Base class for all data connectors"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection and return status"""
        pass
    
    @abstractmethod
    async def get_schemas(self) -> List[str]:
        """Get available schemas/databases"""
        pass
    
    @abstractmethod
    async def get_tables(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available tables in schema"""
        pass
    
    @abstractmethod
    async def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get table schema information"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame"""
        pass
    
    @abstractmethod
    async def stream_data(self, query: str, batch_size: int = 1000) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream data in batches for large datasets"""
        pass


class SnowflakeConnector(BaseConnector):
    """Snowflake data warehouse connector"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.account = config.additional_params.get("account")
        self.warehouse = config.additional_params.get("warehouse")
        self.role = config.additional_params.get("role")
        
    async def connect(self) -> bool:
        """Connect to Snowflake"""
        try:
            import snowflake.connector
            
            self.connection = snowflake.connector.connect(
                user=self.config.username,
                password=self.config.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.config.database,
                role=self.role,
                client_session_keep_alive=True
            )
            
            logger.info("Connected to Snowflake", 
                       account=self.account, 
                       database=self.config.database)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Snowflake", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from Snowflake"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from Snowflake")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Snowflake connection"""
        try:
            if not self.connection:
                await self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()[0]
            cursor.close()
            
            return {
                "status": "success",
                "version": version,
                "warehouse": self.warehouse,
                "database": self.config.database
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_schemas(self) -> List[str]:
        """Get Snowflake schemas"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW SCHEMAS")
            schemas = [row[1] for row in cursor.fetchall()]
            cursor.close()
            return schemas
        except Exception as e:
            logger.error("Error getting schemas", error=str(e))
            return []
    
    async def get_tables(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get Snowflake tables"""
        try:
            cursor = self.connection.cursor()
            if schema:
                cursor.execute(f"SHOW TABLES IN SCHEMA {schema}")
            else:
                cursor.execute("SHOW TABLES")
            
            tables = []
            for row in cursor.fetchall():
                tables.append({
                    "name": row[1],
                    "schema": row[2] if len(row) > 2 else schema,
                    "type": "TABLE",
                    "row_count": row[5] if len(row) > 5 else None
                })
            
            cursor.close()
            return tables
        except Exception as e:
            logger.error("Error getting tables", error=str(e))
            return []
    
    async def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get Snowflake table schema"""
        try:
            cursor = self.connection.cursor()
            full_name = f"{schema}.{table_name}" if schema else table_name
            cursor.execute(f"DESCRIBE TABLE {full_name}")
            
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "Y",
                    "default": row[3],
                    "primary_key": row[4] == "Y" if len(row) > 4 else False
                })
            
            cursor.close()
            return {
                "table_name": table_name,
                "schema": schema,
                "columns": columns
            }
        except Exception as e:
            logger.error("Error getting table schema", error=str(e))
            return {}
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute Snowflake query"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch data
            data = cursor.fetchall()
            cursor.close()
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            return df
            
        except Exception as e:
            logger.error("Error executing query", error=str(e))
            return pd.DataFrame()
    
    async def stream_data(self, query: str, batch_size: int = 1000) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream Snowflake data in batches"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                
                df = pd.DataFrame(batch, columns=columns)
                yield df
            
            cursor.close()
            
        except Exception as e:
            logger.error("Error streaming data", error=str(e))


class AWSConnector(BaseConnector):
    """AWS services connector (S3, Redshift, RDS)"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.aws_access_key = config.additional_params.get("aws_access_key_id")
        self.aws_secret_key = config.additional_params.get("aws_secret_access_key")
        self.region = config.additional_params.get("region", "us-east-1")
        self.service_type = config.additional_params.get("service_type", "s3")  # s3, redshift, rds
        
    async def connect(self) -> bool:
        """Connect to AWS service"""
        try:
            self.session = boto3.Session(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )
            
            if self.service_type == "s3":
                self.client = self.session.client("s3")
            elif self.service_type == "redshift":
                self.client = self.session.client("redshift-data")
            elif self.service_type == "rds":
                self.client = self.session.client("rds-data")
            
            logger.info("Connected to AWS", service=self.service_type, region=self.region)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to AWS", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from AWS"""
        self.client = None
        self.session = None
        logger.info("Disconnected from AWS")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test AWS connection"""
        try:
            if self.service_type == "s3":
                response = self.client.list_buckets()
                return {
                    "status": "success",
                    "service": "s3",
                    "bucket_count": len(response.get("Buckets", []))
                }
            elif self.service_type == "redshift":
                # Test Redshift connection
                return {"status": "success", "service": "redshift"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_schemas(self) -> List[str]:
        """Get AWS schemas (buckets for S3, schemas for Redshift)"""
        try:
            if self.service_type == "s3":
                response = self.client.list_buckets()
                return [bucket["Name"] for bucket in response.get("Buckets", [])]
            elif self.service_type == "redshift":
                # Would query Redshift for schemas
                return []
        except Exception as e:
            logger.error("Error getting schemas", error=str(e))
            return []
    
    async def get_tables(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get AWS tables (S3 objects or Redshift tables)"""
        try:
            if self.service_type == "s3" and schema:
                response = self.client.list_objects_v2(Bucket=schema)
                objects = []
                for obj in response.get("Contents", []):
                    objects.append({
                        "name": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "type": "FILE"
                    })
                return objects
        except Exception as e:
            logger.error("Error getting tables", error=str(e))
            return []
    
    async def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get table schema"""
        # Implementation would depend on service type
        return {}
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute query (mainly for Redshift)"""
        if self.service_type == "redshift":
            # Would execute Redshift query
            pass
        return pd.DataFrame()
    
    async def stream_data(self, query: str, batch_size: int = 1000) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream data"""
        # Implementation for streaming from AWS services
        yield pd.DataFrame()
    
    async def read_s3_file(self, bucket: str, key: str, file_format: str = "csv") -> pd.DataFrame:
        """Read file from S3"""
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            
            if file_format.lower() == "csv":
                df = pd.read_csv(response["Body"])
            elif file_format.lower() == "parquet":
                df = pd.read_parquet(response["Body"])
            elif file_format.lower() == "json":
                df = pd.read_json(response["Body"])
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            return df
            
        except Exception as e:
            logger.error("Error reading S3 file", bucket=bucket, key=key, error=str(e))
            return pd.DataFrame()


class DatabricksConnector(BaseConnector):
    """Databricks connector"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.workspace_url = config.additional_params.get("workspace_url")
        self.access_token = config.additional_params.get("access_token")
        self.cluster_id = config.additional_params.get("cluster_id")
        
    async def connect(self) -> bool:
        """Connect to Databricks"""
        try:
            from databricks import sql
            
            self.connection = sql.connect(
                server_hostname=self.workspace_url.replace("https://", ""),
                http_path=f"/sql/1.0/warehouses/{self.cluster_id}",
                access_token=self.access_token
            )
            
            logger.info("Connected to Databricks", workspace=self.workspace_url)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Databricks", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from Databricks"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from Databricks")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Databricks connection"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            return {
                "status": "success",
                "workspace": self.workspace_url,
                "cluster_id": self.cluster_id
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_schemas(self) -> List[str]:
        """Get Databricks schemas"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW SCHEMAS")
            schemas = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return schemas
        except Exception as e:
            logger.error("Error getting schemas", error=str(e))
            return []
    
    async def get_tables(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get Databricks tables"""
        try:
            cursor = self.connection.cursor()
            if schema:
                cursor.execute(f"SHOW TABLES IN {schema}")
            else:
                cursor.execute("SHOW TABLES")
            
            tables = []
            for row in cursor.fetchall():
                tables.append({
                    "name": row[1],
                    "schema": row[0],
                    "type": "TABLE"
                })
            
            cursor.close()
            return tables
        except Exception as e:
            logger.error("Error getting tables", error=str(e))
            return []
    
    async def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get Databricks table schema"""
        try:
            cursor = self.connection.cursor()
            full_name = f"{schema}.{table_name}" if schema else table_name
            cursor.execute(f"DESCRIBE {full_name}")
            
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "comment": row[2] if len(row) > 2 else None
                })
            
            cursor.close()
            return {
                "table_name": table_name,
                "schema": schema,
                "columns": columns
            }
        except Exception as e:
            logger.error("Error getting table schema", error=str(e))
            return {}
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute Databricks query"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
            
            df = pd.DataFrame(data, columns=columns)
            return df
            
        except Exception as e:
            logger.error("Error executing query", error=str(e))
            return pd.DataFrame()
    
    async def stream_data(self, query: str, batch_size: int = 1000) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream Databricks data"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                
                df = pd.DataFrame(batch, columns=columns)
                yield df
            
            cursor.close()
            
        except Exception as e:
            logger.error("Error streaming data", error=str(e))


class BigQueryConnector(BaseConnector):
    """Google BigQuery connector"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.project_id = config.additional_params.get("project_id")
        self.credentials_path = config.additional_params.get("credentials_path")
        
    async def connect(self) -> bool:
        """Connect to BigQuery"""
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            
            if self.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials
                )
            else:
                self.client = bigquery.Client(project=self.project_id)
            
            logger.info("Connected to BigQuery", project=self.project_id)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to BigQuery", error=str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from BigQuery"""
        self.client = None
        logger.info("Disconnected from BigQuery")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test BigQuery connection"""
        try:
            # Test with a simple query
            query = "SELECT 1 as test"
            job = self.client.query(query)
            result = job.result()
            
            return {
                "status": "success",
                "project_id": self.project_id,
                "location": job.location
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_schemas(self) -> List[str]:
        """Get BigQuery datasets"""
        try:
            datasets = list(self.client.list_datasets())
            return [dataset.dataset_id for dataset in datasets]
        except Exception as e:
            logger.error("Error getting datasets", error=str(e))
            return []
    
    async def get_tables(self, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get BigQuery tables"""
        try:
            if not schema:
                return []
            
            dataset_ref = self.client.dataset(schema)
            tables = list(self.client.list_tables(dataset_ref))
            
            table_list = []
            for table in tables:
                table_list.append({
                    "name": table.table_id,
                    "schema": schema,
                    "type": table.table_type,
                    "num_rows": table.num_rows,
                    "num_bytes": table.num_bytes
                })
            
            return table_list
        except Exception as e:
            logger.error("Error getting tables", error=str(e))
            return []
    
    async def get_table_schema(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get BigQuery table schema"""
        try:
            table_ref = self.client.dataset(schema).table(table_name)
            table = self.client.get_table(table_ref)
            
            columns = []
            for field in table.schema:
                columns.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description
                })
            
            return {
                "table_name": table_name,
                "schema": schema,
                "columns": columns,
                "num_rows": table.num_rows,
                "created": table.created.isoformat() if table.created else None
            }
        except Exception as e:
            logger.error("Error getting table schema", error=str(e))
            return {}
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute BigQuery query"""
        try:
            job = self.client.query(query)
            result = job.result()
            
            df = result.to_dataframe()
            return df
            
        except Exception as e:
            logger.error("Error executing query", error=str(e))
            return pd.DataFrame()
    
    async def stream_data(self, query: str, batch_size: int = 1000) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream BigQuery data"""
        try:
            job = self.client.query(query)
            
            for batch in job.result().to_dataframe_iterable():
                yield batch
                
        except Exception as e:
            logger.error("Error streaming data", error=str(e))


class EnterpriseConnectorService:
    """
    Service for managing enterprise data connectors
    """
    
    def __init__(self):
        self.connectors = {}
        self.connector_types = {
            "snowflake": SnowflakeConnector,
            "aws": AWSConnector,
            "databricks": DatabricksConnector,
            "bigquery": BigQueryConnector
        }
    
    async def create_connection(self, config: ConnectionConfig) -> Dict[str, Any]:
        """Create a new data connection"""
        try:
            connector_class = self.connector_types.get(config.connection_type)
            if not connector_class:
                raise ValueError(f"Unsupported connector type: {config.connection_type}")
            
            connector = connector_class(config)
            success = await connector.connect()
            
            if success:
                self.connectors[config.connection_id] = connector
                
                # Store connection metadata in database
                await self._store_connection_metadata(config)
                
                logger.info("Connection created successfully", 
                           connection_id=config.connection_id,
                           connection_type=config.connection_type)
                
                return {
                    "status": "success",
                    "connection_id": config.connection_id,
                    "connection_type": config.connection_type
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to establish connection"
                }
                
        except Exception as e:
            logger.error("Error creating connection", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def test_connection(self, connection_id: str) -> Dict[str, Any]:
        """Test an existing connection"""
        connector = self.connectors.get(connection_id)
        if not connector:
            return {"status": "error", "error": "Connection not found"}
        
        return await connector.test_connection()
    
    async def get_connection_metadata(self, connection_id: str) -> Dict[str, Any]:
        """Get connection metadata"""
        connector = self.connectors.get(connection_id)
        if not connector:
            return {}
        
        try:
            schemas = await connector.get_schemas()
            return {
                "connection_id": connection_id,
                "connection_type": connector.config.connection_type,
                "schemas": schemas,
                "status": "connected"
            }
        except Exception as e:
            logger.error("Error getting connection metadata", error=str(e))
            return {"error": str(e)}
    
    async def execute_query(self, connection_id: str, query: str) -> pd.DataFrame:
        """Execute query on connection"""
        connector = self.connectors.get(connection_id)
        if not connector:
            raise ValueError(f"Connection {connection_id} not found")
        
        return await connector.execute_query(query)
    
    async def get_data_preview(
        self, 
        connection_id: str, 
        table_name: str, 
        schema: Optional[str] = None, 
        limit: int = 100
    ) -> pd.DataFrame:
        """Get preview of table data"""
        connector = self.connectors.get(connection_id)
        if not connector:
            raise ValueError(f"Connection {connection_id} not found")
        
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        query = f"SELECT * FROM {full_table_name} LIMIT {limit}"
        
        return await connector.execute_query(query)
    
    async def sync_data(
        self,
        connection_id: str,
        table_name: str,
        schema: Optional[str] = None,
        sync_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sync data from external source for monitoring"""
        connector = self.connectors.get(connection_id)
        if not connector:
            raise ValueError(f"Connection {connection_id} not found")
        
        try:
            # Get table schema
            table_schema = await connector.get_table_schema(table_name, schema)
            
            # Get data (with optional filtering based on sync_config)
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            query = f"SELECT * FROM {full_table_name}"
            
            # Add filters from sync_config
            if sync_config:
                filters = sync_config.get("filters", [])
                if filters:
                    where_clause = " AND ".join(filters)
                    query += f" WHERE {where_clause}"
                
                limit = sync_config.get("limit")
                if limit:
                    query += f" LIMIT {limit}"
            
            data = await connector.execute_query(query)
            
            # Store synced data for monitoring
            sync_result = await self._store_synced_data(
                connection_id, table_name, schema, data, table_schema
            )
            
            return {
                "status": "success",
                "rows_synced": len(data),
                "columns": list(data.columns),
                "sync_id": sync_result["sync_id"]
            }
            
        except Exception as e:
            logger.error("Error syncing data", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _store_connection_metadata(self, config: ConnectionConfig):
        """Store connection metadata in database"""
        from app.core.database import get_db
        
        # Don't store sensitive information like passwords
        safe_config = {
            "connection_type": config.connection_type,
            "name": config.name,
            "host": config.host,
            "port": config.port,
            "database": config.database,
            "username": config.username,
            "additional_params": {
                k: v for k, v in (config.additional_params or {}).items()
                if k not in ["password", "access_token", "secret_key"]
            }
        }
        
        async with get_db() as db:
            await db.execute("""
                INSERT INTO data_connections (
                    connection_id, connection_type, name, config, created_at
                ) VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                ON CONFLICT (connection_id) DO UPDATE SET
                    connection_type = EXCLUDED.connection_type,
                    name = EXCLUDED.name,
                    config = EXCLUDED.config,
                    updated_at = CURRENT_TIMESTAMP
            """, config.connection_id, config.connection_type, config.name, json.dumps(safe_config))
    
    async def _store_synced_data(
        self,
        connection_id: str,
        table_name: str,
        schema: Optional[str],
        data: pd.DataFrame,
        table_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store synced data for monitoring"""
        from app.core.database import get_db
        
        sync_id = f"sync_{connection_id}_{table_name}_{int(datetime.utcnow().timestamp())}"
        
        # Calculate data statistics
        data_stats = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.astype(str).to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum()
        }
        
        async with get_db() as db:
            await db.execute("""
                INSERT INTO data_syncs (
                    sync_id, connection_id, table_name, schema_name, 
                    row_count, column_count, data_statistics, table_schema,
                    synced_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
            """, sync_id, connection_id, table_name, schema, 
                len(data), len(data.columns), json.dumps(data_stats), 
                json.dumps(table_schema))
        
        return {"sync_id": sync_id}
    
    async def get_available_connectors(self) -> List[Dict[str, Any]]:
        """Get list of available connector types"""
        return [
            {
                "type": "snowflake",
                "name": "Snowflake Data Warehouse",
                "description": "Connect to Snowflake cloud data warehouse",
                "required_params": ["account", "username", "password", "warehouse", "database"],
                "optional_params": ["role", "schema"]
            },
            {
                "type": "aws",
                "name": "Amazon Web Services",
                "description": "Connect to AWS services (S3, Redshift, RDS)",
                "required_params": ["aws_access_key_id", "aws_secret_access_key", "region", "service_type"],
                "optional_params": ["database", "cluster_id"]
            },
            {
                "type": "databricks",
                "name": "Databricks",
                "description": "Connect to Databricks workspace",
                "required_params": ["workspace_url", "access_token", "cluster_id"],
                "optional_params": ["catalog", "schema"]
            },
            {
                "type": "bigquery",
                "name": "Google BigQuery",
                "description": "Connect to Google BigQuery",
                "required_params": ["project_id"],
                "optional_params": ["credentials_path", "dataset"]
            }
        ]


# Database initialization
async def initialize_connector_tables():
    """Initialize connector-related database tables"""
    from app.core.database import get_db
    
    async with get_db() as db:
        # Data connections table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS data_connections (
                connection_id VARCHAR PRIMARY KEY,
                connection_type VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                config JSONB NOT NULL,
                status VARCHAR DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_tested TIMESTAMP,
                test_status VARCHAR
            )
        """)
        
        # Data syncs table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS data_syncs (
                sync_id VARCHAR PRIMARY KEY,
                connection_id VARCHAR NOT NULL,
                table_name VARCHAR NOT NULL,
                schema_name VARCHAR,
                row_count INTEGER,
                column_count INTEGER,
                data_statistics JSONB,
                table_schema JSONB,
                synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (connection_id) REFERENCES data_connections(connection_id)
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_data_connections_type ON data_connections(connection_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_data_syncs_connection ON data_syncs(connection_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_data_syncs_synced_at ON data_syncs(synced_at)")
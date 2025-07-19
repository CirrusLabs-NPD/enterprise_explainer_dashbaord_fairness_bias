"""
Data Management Service
Handles data upload, preprocessing, and quality assessment
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import structlog
from io import StringIO, BytesIO

# Data processing libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import ydata_profiling

from app.models.model_metadata import DatasetMetadata, DataQualityReport
from app.core.worker_pool import WorkerPool, TaskType
from app.core.database import get_db
from app.config import settings

logger = structlog.get_logger(__name__)


class DataService:
    """
    Comprehensive data management service
    
    Features:
    - Data upload and storage
    - Data preprocessing pipelines
    - Quality assessment
    - Statistical analysis
    - Data streaming integration
    """
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.dataset_cache: Dict[str, pd.DataFrame] = {}
        self.metadata_cache: Dict[str, DatasetMetadata] = {}
        
        # Ensure data directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        logger.info("DataService initialized")
    
    async def upload_dataset(
        self,
        file_content: bytes,
        filename: str,
        dataset_name: str,
        description: str = "",
        uploaded_by: str = "system"
    ) -> str:
        """
        Upload and store a dataset
        """
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(dataset_name, file_content)
            
            # Determine file format
            file_format = self._detect_file_format(filename)
            
            # Save file
            file_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_id}.{file_format}")
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Load and analyze data
            df = await self._load_dataframe(file_path, file_format)
            
            # Create metadata
            metadata = DatasetMetadata(
                dataset_id=dataset_id,
                name=dataset_name,
                description=description,
                file_path=file_path,
                format=file_format,
                size_bytes=len(file_content),
                num_rows=len(df),
                num_columns=len(df.columns),
                column_names=df.columns.tolist(),
                column_types=df.dtypes.astype(str).to_dict(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save metadata
            await self._save_dataset_metadata(metadata)
            
            # Cache data and metadata
            self.dataset_cache[dataset_id] = df
            self.metadata_cache[dataset_id] = metadata
            
            logger.info(f"Dataset {dataset_id} uploaded successfully")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Error uploading dataset: {e}")
            raise
    
    async def get_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Get dataset metadata
        """
        if dataset_id in self.metadata_cache:
            return self.metadata_cache[dataset_id]
        
        try:
            async with get_db() as db:
                query = "SELECT * FROM dataset_metadata WHERE dataset_id = :dataset_id"
                result = await db.fetch_one(query, {"dataset_id": dataset_id})
                
                if result:
                    metadata = DatasetMetadata(**dict(result))
                    self.metadata_cache[dataset_id] = metadata
                    return metadata
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting dataset metadata: {e}")
            return None
    
    async def list_datasets(self) -> List[DatasetMetadata]:
        """
        List all datasets
        """
        try:
            async with get_db() as db:
                query = "SELECT * FROM dataset_metadata ORDER BY created_at DESC"
                results = await db.fetch_all(query)
                
                return [DatasetMetadata(**dict(result)) for result in results]
                
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset
        """
        try:
            # Get metadata
            metadata = await self.get_dataset_metadata(dataset_id)
            if not metadata:
                return False
            
            # Delete file
            if os.path.exists(metadata.file_path):
                os.remove(metadata.file_path)
            
            # Delete from database
            async with get_db() as db:
                query = "DELETE FROM dataset_metadata WHERE dataset_id = :dataset_id"
                await db.execute(query, {"dataset_id": dataset_id})
            
            # Remove from caches
            self.dataset_cache.pop(dataset_id, None)
            self.metadata_cache.pop(dataset_id, None)
            
            logger.info(f"Dataset {dataset_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False
    
    async def get_dataset_sample(self, dataset_id: str, n_rows: int = 100) -> List[Dict]:
        """
        Get a sample of the dataset
        """
        try:
            df = await self._load_dataset(dataset_id)
            sample = df.head(n_rows)
            return sample.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting dataset sample: {e}")
            raise
    
    async def compute_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics
        """
        try:
            # Submit statistics computation task
            task_id = f"stats_{dataset_id}"
            
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._compute_statistics_worker,
                args=(dataset_id,),
                task_type=TaskType.CPU_INTENSIVE
            )
            
            result = await self.worker_pool.get_result(task_id)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Statistics computation failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            raise
    
    async def assess_data_quality(self, dataset_id: str) -> DataQualityReport:
        """
        Assess data quality
        """
        try:
            # Submit quality assessment task
            task_id = f"quality_{dataset_id}"
            
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._assess_quality_worker,
                args=(dataset_id,),
                task_type=TaskType.CPU_INTENSIVE
            )
            
            result = await self.worker_pool.get_result(task_id)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Quality assessment failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            raise
    
    async def split_dataset(
        self,
        dataset_id: str,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, str]:
        """
        Split dataset into train/validation/test sets
        """
        try:
            df = await self._load_dataset(dataset_id)
            
            # First split: train + validation vs test
            train_val, test = train_test_split(
                df, 
                test_size=test_ratio, 
                random_state=random_state
            )
            
            # Second split: train vs validation
            val_size = validation_ratio / (train_ratio + validation_ratio)
            train, validation = train_test_split(
                train_val, 
                test_size=val_size, 
                random_state=random_state
            )
            
            # Save splits
            splits = {}
            for split_name, split_df in [("train", train), ("validation", validation), ("test", test)]:
                split_id = f"{dataset_id}_{split_name}"
                split_path = os.path.join(settings.UPLOAD_DIR, f"{split_id}.parquet")
                split_df.to_parquet(split_path)
                
                # Create metadata for split
                metadata = DatasetMetadata(
                    dataset_id=split_id,
                    name=f"{dataset_id}_{split_name}",
                    description=f"{split_name} split of {dataset_id}",
                    file_path=split_path,
                    format="parquet",
                    size_bytes=os.path.getsize(split_path),
                    num_rows=len(split_df),
                    num_columns=len(split_df.columns),
                    column_names=split_df.columns.tolist(),
                    column_types=split_df.dtypes.astype(str).to_dict(),
                    tags=[f"split:{split_name}", f"parent:{dataset_id}"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                await self._save_dataset_metadata(metadata)
                splits[split_name] = split_id
            
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            raise
    
    async def apply_preprocessing(
        self,
        dataset_id: str,
        preprocessing_steps: List[Dict[str, Any]]
    ) -> str:
        """
        Apply preprocessing steps to dataset
        """
        try:
            # Submit preprocessing task
            task_id = f"preprocess_{dataset_id}"
            
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._apply_preprocessing_worker,
                args=(dataset_id, preprocessing_steps),
                task_type=TaskType.CPU_INTENSIVE
            )
            
            result = await self.worker_pool.get_result(task_id)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Preprocessing failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error applying preprocessing: {e}")
            raise
    
    async def profile_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive data profile
        """
        try:
            # Submit profiling task
            task_id = f"profile_{dataset_id}"
            
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._profile_dataset_worker,
                args=(dataset_id,),
                task_type=TaskType.CPU_INTENSIVE
            )
            
            result = await self.worker_pool.get_result(task_id)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Profiling failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error profiling dataset: {e}")
            raise
    
    async def validate_dataset(
        self,
        dataset_id: str,
        validation_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate dataset against rules
        """
        try:
            # Submit validation task
            task_id = f"validate_{dataset_id}"
            
            await self.worker_pool.submit_task(
                task_id=task_id,
                function=self._validate_dataset_worker,
                args=(dataset_id, validation_rules),
                task_type=TaskType.CPU_INTENSIVE
            )
            
            result = await self.worker_pool.get_result(task_id)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"Validation failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            raise
    
    async def connect_stream(self, stream_config: Dict[str, Any]) -> str:
        """
        Connect to a data stream
        """
        try:
            # Generate stream ID
            stream_id = f"stream_{int(datetime.utcnow().timestamp())}"
            
            # This would implement actual stream connection
            # For now, return a placeholder
            logger.info(f"Stream {stream_id} connected")
            
            return stream_id
            
        except Exception as e:
            logger.error(f"Error connecting to stream: {e}")
            raise
    
    async def disconnect_stream(self, stream_id: str) -> bool:
        """
        Disconnect from a data stream
        """
        try:
            # This would implement actual stream disconnection
            logger.info(f"Stream {stream_id} disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting stream: {e}")
            return False
    
    async def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """
        Get stream status
        """
        try:
            # This would return actual stream status
            return {
                "stream_id": stream_id,
                "status": "connected",
                "messages_processed": 1000,
                "last_message": "2024-01-15T10:30:00Z"
            }
            
        except Exception as e:
            logger.error(f"Error getting stream status: {e}")
            raise
    
    # Worker functions
    
    @staticmethod
    def _compute_statistics_worker(dataset_id: str) -> Dict[str, Any]:
        """Worker function for computing statistics"""
        try:
            # Load dataset
            df = DataService._load_dataset_sync(dataset_id)
            
            # Compute basic statistics
            stats = {
                "shape": df.shape,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicates": df.duplicated().sum(),
                "memory_usage": df.memory_usage(deep=True).sum(),
            }
            
            # Numeric columns statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns statistics
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                stats["categorical_stats"] = {}
                for col in categorical_cols:
                    stats["categorical_stats"][col] = {
                        "unique_values": df[col].nunique(),
                        "top_values": df[col].value_counts().head(10).to_dict()
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in statistics worker: {e}")
            raise
    
    @staticmethod
    def _assess_quality_worker(dataset_id: str) -> DataQualityReport:
        """Worker function for quality assessment"""
        try:
            # Load dataset
            df = DataService._load_dataset_sync(dataset_id)
            
            # Basic quality metrics
            total_rows, total_features = df.shape
            missing_values = df.isnull().sum().to_dict()
            duplicate_rows = df.duplicated().sum()
            
            # Detect outliers (using IQR method)
            outliers = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outliers[col] = outlier_count
            
            # Data types
            data_types = df.dtypes.astype(str).to_dict()
            
            # Calculate quality score
            quality_score = DataService._calculate_quality_score(
                df, missing_values, duplicate_rows, outliers
            )
            
            # Identify issues
            issues = []
            if sum(missing_values.values()) > 0:
                issues.append(f"Missing values detected in {len([k for k, v in missing_values.items() if v > 0])} columns")
            if duplicate_rows > 0:
                issues.append(f"{duplicate_rows} duplicate rows detected")
            if sum(outliers.values()) > total_rows * 0.1:
                issues.append("High number of outliers detected")
            
            return DataQualityReport(
                dataset_id=dataset_id,
                total_rows=total_rows,
                total_features=total_features,
                missing_values=missing_values,
                duplicate_rows=duplicate_rows,
                outliers=outliers,
                data_types=data_types,
                quality_score=quality_score,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Error in quality assessment worker: {e}")
            raise
    
    @staticmethod
    def _apply_preprocessing_worker(
        dataset_id: str,
        preprocessing_steps: List[Dict[str, Any]]
    ) -> str:
        """Worker function for preprocessing"""
        try:
            # Load dataset
            df = DataService._load_dataset_sync(dataset_id)
            
            # Apply preprocessing steps
            for step in preprocessing_steps:
                step_type = step.get("type")
                params = step.get("params", {})
                
                if step_type == "drop_columns":
                    columns = params.get("columns", [])
                    df = df.drop(columns=columns)
                
                elif step_type == "fill_missing":
                    strategy = params.get("strategy", "mean")
                    columns = params.get("columns", df.columns.tolist())
                    
                    if strategy in ["mean", "median", "most_frequent"]:
                        imputer = SimpleImputer(strategy=strategy)
                        df[columns] = imputer.fit_transform(df[columns])
                
                elif step_type == "encode_categorical":
                    columns = params.get("columns", [])
                    method = params.get("method", "onehot")
                    
                    if method == "onehot":
                        df = pd.get_dummies(df, columns=columns)
                    elif method == "label":
                        for col in columns:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col])
                
                elif step_type == "scale_numeric":
                    columns = params.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
                    method = params.get("method", "standard")
                    
                    if method == "standard":
                        scaler = StandardScaler()
                        df[columns] = scaler.fit_transform(df[columns])
            
            # Save processed dataset
            processed_id = f"{dataset_id}_processed"
            processed_path = os.path.join(settings.UPLOAD_DIR, f"{processed_id}.parquet")
            df.to_parquet(processed_path)
            
            return processed_id
            
        except Exception as e:
            logger.error(f"Error in preprocessing worker: {e}")
            raise
    
    @staticmethod
    def _profile_dataset_worker(dataset_id: str) -> Dict[str, Any]:
        """Worker function for profiling"""
        try:
            # Load dataset
            df = DataService._load_dataset_sync(dataset_id)
            
            # Generate profile (simplified version)
            profile = {
                "overview": {
                    "n_rows": len(df),
                    "n_columns": len(df.columns),
                    "missing_cells": df.isnull().sum().sum(),
                    "duplicated_rows": df.duplicated().sum(),
                    "memory_size": df.memory_usage(deep=True).sum()
                },
                "variables": {}
            }
            
            # Profile each column
            for col in df.columns:
                col_profile = {
                    "type": str(df[col].dtype),
                    "missing_count": df[col].isnull().sum(),
                    "unique_count": df[col].nunique(),
                }
                
                if df[col].dtype in [np.number]:
                    col_profile.update({
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "median": df[col].median(),
                    })
                else:
                    col_profile.update({
                        "top_values": df[col].value_counts().head(10).to_dict()
                    })
                
                profile["variables"][col] = col_profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error in profiling worker: {e}")
            raise
    
    @staticmethod
    def _validate_dataset_worker(
        dataset_id: str,
        validation_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Worker function for dataset validation"""
        try:
            # Load dataset
            df = DataService._load_dataset_sync(dataset_id)
            
            validation_result = {
                "dataset_id": dataset_id,
                "is_valid": True,
                "violations": [],
                "summary": {}
            }
            
            # Apply validation rules
            for rule in validation_rules:
                rule_type = rule.get("type")
                params = rule.get("params", {})
                
                if rule_type == "required_columns":
                    required_cols = params.get("columns", [])
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        validation_result["is_valid"] = False
                        validation_result["violations"].append({
                            "rule": "required_columns",
                            "message": f"Missing required columns: {missing_cols}"
                        })
                
                elif rule_type == "no_missing_values":
                    columns = params.get("columns", df.columns.tolist())
                    for col in columns:
                        if df[col].isnull().any():
                            validation_result["is_valid"] = False
                            validation_result["violations"].append({
                                "rule": "no_missing_values",
                                "message": f"Column {col} has missing values"
                            })
                
                elif rule_type == "data_type":
                    column = params.get("column")
                    expected_type = params.get("expected_type")
                    if column in df.columns and str(df[column].dtype) != expected_type:
                        validation_result["is_valid"] = False
                        validation_result["violations"].append({
                            "rule": "data_type",
                            "message": f"Column {column} has type {df[column].dtype}, expected {expected_type}"
                        })
            
            # Summary statistics
            validation_result["summary"] = {
                "total_rules": len(validation_rules),
                "violations": len(validation_result["violations"]),
                "success_rate": (len(validation_rules) - len(validation_result["violations"])) / len(validation_rules) if validation_rules else 1.0
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in validation worker: {e}")
            raise
    
    # Utility methods
    
    def _generate_dataset_id(self, dataset_name: str, content: bytes) -> str:
        """Generate unique dataset ID"""
        content_hash = hashlib.md5(content).hexdigest()[:8]
        timestamp = int(datetime.utcnow().timestamp())
        return f"{dataset_name}_{timestamp}_{content_hash}"
    
    def _detect_file_format(self, filename: str) -> str:
        """Detect file format from filename"""
        if filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith('.parquet'):
            return 'parquet'
        elif filename.endswith('.json'):
            return 'json'
        else:
            return 'csv'  # Default
    
    async def _load_dataframe(self, file_path: str, file_format: str) -> pd.DataFrame:
        """Load dataframe from file"""
        try:
            if file_format == 'csv':
                return pd.read_csv(file_path)
            elif file_format == 'parquet':
                return pd.read_parquet(file_path)
            elif file_format == 'json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        except Exception as e:
            logger.error(f"Error loading dataframe: {e}")
            raise
    
    async def _load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset with caching"""
        if dataset_id in self.dataset_cache:
            return self.dataset_cache[dataset_id]
        
        metadata = await self.get_dataset_metadata(dataset_id)
        if not metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        df = await self._load_dataframe(metadata.file_path, metadata.format)
        self.dataset_cache[dataset_id] = df
        
        return df
    
    @staticmethod
    def _load_dataset_sync(dataset_id: str) -> pd.DataFrame:
        """Synchronous dataset loading for worker processes"""
        # This would implement synchronous loading
        # For now, return a dummy DataFrame
        return pd.DataFrame(np.random.randn(1000, 5), columns=['A', 'B', 'C', 'D', 'E'])
    
    @staticmethod
    def _calculate_quality_score(
        df: pd.DataFrame,
        missing_values: Dict[str, int],
        duplicate_rows: int,
        outliers: Dict[str, int]
    ) -> float:
        """Calculate overall data quality score"""
        total_rows = len(df)
        total_cells = df.size
        
        # Missing values score
        missing_score = 1.0 - (sum(missing_values.values()) / total_cells)
        
        # Duplicate rows score
        duplicate_score = 1.0 - (duplicate_rows / total_rows)
        
        # Outliers score
        outlier_score = 1.0 - (sum(outliers.values()) / total_rows)
        
        # Weighted average
        quality_score = (missing_score * 0.4 + duplicate_score * 0.3 + outlier_score * 0.3)
        
        return max(0.0, min(1.0, quality_score))
    
    async def _save_dataset_metadata(self, metadata: DatasetMetadata):
        """Save dataset metadata to database"""
        try:
            async with get_db() as db:
                query = """
                INSERT INTO dataset_metadata (
                    dataset_id, name, description, file_path, format, size_bytes,
                    num_rows, num_columns, column_names, column_types, statistics,
                    created_at, updated_at, tags
                ) VALUES (
                    :dataset_id, :name, :description, :file_path, :format, :size_bytes,
                    :num_rows, :num_columns, :column_names, :column_types, :statistics,
                    :created_at, :updated_at, :tags
                )
                ON CONFLICT (dataset_id) DO UPDATE SET
                    name = :name,
                    description = :description,
                    file_path = :file_path,
                    format = :format,
                    size_bytes = :size_bytes,
                    num_rows = :num_rows,
                    num_columns = :num_columns,
                    column_names = :column_names,
                    column_types = :column_types,
                    statistics = :statistics,
                    updated_at = :updated_at,
                    tags = :tags
                """
                
                await db.execute(query, {
                    "dataset_id": metadata.dataset_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "file_path": metadata.file_path,
                    "format": metadata.format,
                    "size_bytes": metadata.size_bytes,
                    "num_rows": metadata.num_rows,
                    "num_columns": metadata.num_columns,
                    "column_names": json.dumps(metadata.column_names),
                    "column_types": json.dumps(metadata.column_types),
                    "statistics": json.dumps(metadata.statistics),
                    "created_at": metadata.created_at,
                    "updated_at": metadata.updated_at,
                    "tags": json.dumps(metadata.tags)
                })
                
        except Exception as e:
            logger.error(f"Error saving dataset metadata: {e}")
            raise
"""
Tests for validation and sanitization
"""

import pytest
import tempfile
import pandas as pd
from io import StringIO
from fastapi import UploadFile

from app.core.validation import (
    FileValidator, DataValidator, ModelValidator, InputSanitizer,
    ModelMetadataRequest, DataUploadRequest
)
from app.core.exceptions import ValidationError


class TestFileValidator:
    """Test file validation functionality"""
    
    def test_validate_file_type_valid(self):
        """Test file type validation with valid files"""
        assert FileValidator.validate_file_type("model.pkl", "model")
        assert FileValidator.validate_file_type("data.csv", "data")
        assert FileValidator.validate_file_type("image.jpg", "image")
        assert FileValidator.validate_file_type("doc.pdf", "document")
    
    def test_validate_file_type_invalid(self):
        """Test file type validation with invalid files"""
        assert not FileValidator.validate_file_type("model.txt", "model")
        assert not FileValidator.validate_file_type("data.exe", "data")
        assert not FileValidator.validate_file_type("", "image")
        assert not FileValidator.validate_file_type("file_without_extension", "document")
    
    def test_validate_file_size(self):
        """Test file size validation"""
        # 1MB file should be valid for most categories
        assert FileValidator.validate_file_size(1024 * 1024, "image")
        assert FileValidator.validate_file_size(1024 * 1024, "document")
        
        # 600MB file should be invalid for all categories
        assert not FileValidator.validate_file_size(600 * 1024 * 1024, "model")
        assert not FileValidator.validate_file_size(600 * 1024 * 1024, "data")
    
    @pytest.mark.asyncio
    async def test_validate_upload_file_valid_csv(self, mock_csv_file):
        """Test upload file validation with valid CSV"""
        with open(mock_csv_file, 'rb') as f:
            content = f.read()
        
        # Create mock UploadFile
        file = UploadFile(
            filename="test.csv",
            file=StringIO(content.decode('utf-8')),
            size=len(content)
        )
        
        result = await FileValidator.validate_upload_file(file, "data")
        
        assert result["valid"]
        assert result["filename"] == "test.csv"
        assert result["rows"] == 5
        assert result["columns"] == 4
        assert "feature1" in result["column_names"]
    
    @pytest.mark.asyncio
    async def test_validate_upload_file_invalid_type(self):
        """Test upload file validation with invalid file type"""
        content = b"invalid content"
        file = UploadFile(
            filename="test.exe",
            file=StringIO(content.decode('utf-8')),
            size=len(content)
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await FileValidator.validate_upload_file(file, "data")
        
        assert "Invalid file type" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_upload_file_too_large(self):
        """Test upload file validation with file too large"""
        # Create content larger than allowed
        large_content = b"x" * (200 * 1024 * 1024)  # 200MB
        file = UploadFile(
            filename="test.csv",
            file=StringIO(large_content.decode('utf-8', errors='ignore')),
            size=len(large_content)
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await FileValidator.validate_upload_file(file, "data")
        
        assert "File too large" in str(exc_info.value)


class TestDataValidator:
    """Test data validation functionality"""
    
    def test_validate_dataframe_valid(self):
        """Test DataFrame validation with valid data"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2.0, 3.0, 4.0, 5.0, 6.0],
            'target': [0, 1, 0, 1, 0]
        })
        
        result = DataValidator.validate_dataframe(
            df,
            required_columns=['feature1', 'target'],
            numeric_columns=['feature1', 'feature2']
        )
        
        assert result["valid"]
        assert result["info"]["rows"] == 5
        assert result["info"]["columns"] == 3
        assert len(result["errors"]) == 0
    
    def test_validate_dataframe_missing_columns(self):
        """Test DataFrame validation with missing required columns"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2.0, 3.0, 4.0]
        })
        
        result = DataValidator.validate_dataframe(
            df,
            required_columns=['feature1', 'target']
        )
        
        assert not result["valid"]
        assert any("Missing required columns" in error for error in result["errors"])
    
    def test_validate_dataframe_non_numeric_columns(self):
        """Test DataFrame validation with non-numeric columns"""
        df = pd.DataFrame({
            'feature1': ['a', 'b', 'c'],
            'feature2': [2.0, 3.0, 4.0],
            'target': [0, 1, 0]
        })
        
        result = DataValidator.validate_dataframe(
            df,
            numeric_columns=['feature1', 'feature2']
        )
        
        assert not result["valid"]
        assert any("must be numeric" in error for error in result["errors"])
    
    def test_validate_dataframe_with_missing_values(self):
        """Test DataFrame validation with missing values"""
        df = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'feature2': [2.0, None, 4.0, 5.0, 6.0],
            'target': [0, 1, 0, 1, 0]
        })
        
        result = DataValidator.validate_dataframe(df)
        
        assert result["valid"]  # Missing values don't make it invalid, just warnings
        assert len(result["warnings"]) > 0
        assert any(w["type"] == "missing_values" for w in result["warnings"])
    
    def test_validate_dataframe_with_duplicates(self):
        """Test DataFrame validation with duplicate rows"""
        df = pd.DataFrame({
            'feature1': [1, 2, 1, 4, 5],
            'feature2': [2.0, 3.0, 2.0, 5.0, 6.0],
            'target': [0, 1, 0, 1, 0]
        })
        
        result = DataValidator.validate_dataframe(df)
        
        assert result["valid"]
        assert any(w["type"] == "duplicate_rows" for w in result["warnings"])


class TestModelValidator:
    """Test model validation functionality"""
    
    def test_validate_model_metadata_valid(self):
        """Test model metadata validation with valid data"""
        metadata = {
            "name": "Test Model",
            "model_type": "classification",
            "feature_names": ["feature1", "feature2", "feature3"],
            "description": "A test model",
            "version": "1.0.0",
            "framework": "sklearn"
        }
        
        result = ModelValidator.validate_model_metadata(metadata)
        
        assert result["valid"]
        assert len(result["errors"]) == 0
    
    def test_validate_model_metadata_missing_required(self):
        """Test model metadata validation with missing required fields"""
        metadata = {
            "description": "A test model"
        }
        
        result = ModelValidator.validate_model_metadata(metadata)
        
        assert not result["valid"]
        assert any("Missing required field: name" in error for error in result["errors"])
        assert any("Missing required field: model_type" in error for error in result["errors"])
        assert any("Missing required field: feature_names" in error for error in result["errors"])
    
    def test_validate_model_metadata_invalid_type(self):
        """Test model metadata validation with invalid model type"""
        metadata = {
            "name": "Test Model",
            "model_type": "invalid_type",
            "feature_names": ["feature1", "feature2"]
        }
        
        result = ModelValidator.validate_model_metadata(metadata)
        
        assert not result["valid"]
        assert any("Invalid model_type" in error for error in result["errors"])
    
    def test_validate_model_metadata_empty_features(self):
        """Test model metadata validation with empty feature names"""
        metadata = {
            "name": "Test Model",
            "model_type": "classification",
            "feature_names": []
        }
        
        result = ModelValidator.validate_model_metadata(metadata)
        
        assert not result["valid"]
        assert any("feature_names cannot be empty" in error for error in result["errors"])


class TestInputSanitizer:
    """Test input sanitization functionality"""
    
    def test_sanitize_string_basic(self):
        """Test basic string sanitization"""
        result = InputSanitizer.sanitize_string("  Hello World  ")
        assert result == "Hello World"
        
        result = InputSanitizer.sanitize_string("Test\x00String")
        assert result == "TestString"
    
    def test_sanitize_string_length_limit(self):
        """Test string length limitation"""
        long_string = "a" * 2000
        result = InputSanitizer.sanitize_string(long_string, max_length=100)
        assert len(result) == 100
    
    def test_sanitize_string_html_removal(self):
        """Test HTML tag removal"""
        html_string = "<script>alert('xss')</script>Hello<b>World</b>"
        result = InputSanitizer.sanitize_string(html_string, allow_html=False)
        assert result == "alert('xss')HelloWorld"
        
        result = InputSanitizer.sanitize_string(html_string, allow_html=True)
        assert "<script>" in result
    
    def test_sanitize_string_special_chars(self):
        """Test special character handling"""
        special_string = "Hello@#$%^&*()World!"
        result = InputSanitizer.sanitize_string(special_string, allow_special_chars=False)
        assert result == "HelloWorld"
        
        result = InputSanitizer.sanitize_string(special_string, allow_special_chars=True)
        assert "@#$%^&*()" in result
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        dangerous_filename = "<>:\"|?*script.txt"
        result = InputSanitizer.sanitize_filename(dangerous_filename)
        assert result == "script.txt"
        
        path_filename = "../../../etc/passwd"
        result = InputSanitizer.sanitize_filename(path_filename)
        assert result == "..etcpasswd"
        
        long_filename = "a" * 300 + ".txt"
        result = InputSanitizer.sanitize_filename(long_filename)
        assert len(result) <= 255
        assert result.endswith(".txt")
    
    def test_sanitize_json_valid(self):
        """Test JSON sanitization with valid JSON"""
        json_string = '{"key": "value", "number": 123}'
        result = InputSanitizer.sanitize_json(json_string)
        
        assert result["key"] == "value"
        assert result["number"] == 123
    
    def test_sanitize_json_invalid(self):
        """Test JSON sanitization with invalid JSON"""
        invalid_json = '{"key": "value"'  # Missing closing brace
        
        with pytest.raises(ValidationError) as exc_info:
            InputSanitizer.sanitize_json(invalid_json)
        
        assert "Invalid JSON format" in str(exc_info.value)
    
    def test_sanitize_json_too_deep(self):
        """Test JSON sanitization with too much nesting"""
        # Create deeply nested JSON
        nested_json = "{" + "\"key\":{" * 15 + "\"value\":1" + "}" * 16
        
        with pytest.raises(ValidationError) as exc_info:
            InputSanitizer.sanitize_json(nested_json, max_depth=10)
        
        assert "too deep" in str(exc_info.value)


class TestPydanticModels:
    """Test Pydantic validation models"""
    
    def test_model_metadata_request_valid(self):
        """Test valid model metadata request"""
        data = {
            "name": "Test Model",
            "model_type": "classification",
            "feature_names": ["feature1", "feature2", "feature3"],
            "description": "A test model",
            "version": "1.0.0"
        }
        
        model = ModelMetadataRequest(**data)
        assert model.name == "Test Model"
        assert model.model_type == "classification"
        assert len(model.feature_names) == 3
    
    def test_model_metadata_request_invalid_type(self):
        """Test invalid model type in metadata request"""
        data = {
            "name": "Test Model",
            "model_type": "invalid_type",
            "feature_names": ["feature1", "feature2"]
        }
        
        with pytest.raises(ValueError) as exc_info:
            ModelMetadataRequest(**data)
        
        assert "Model type must be one of" in str(exc_info.value)
    
    def test_model_metadata_request_empty_name(self):
        """Test empty name in metadata request"""
        data = {
            "name": "",
            "model_type": "classification",
            "feature_names": ["feature1", "feature2"]
        }
        
        with pytest.raises(ValueError) as exc_info:
            ModelMetadataRequest(**data)
        
        assert "Name cannot be empty" in str(exc_info.value)
    
    def test_model_metadata_request_duplicate_features(self):
        """Test duplicate feature names"""
        data = {
            "name": "Test Model",
            "model_type": "classification",
            "feature_names": ["feature1", "feature2", "feature1"]
        }
        
        with pytest.raises(ValueError) as exc_info:
            ModelMetadataRequest(**data)
        
        assert "Feature names must be unique" in str(exc_info.value)
    
    def test_data_upload_request_valid(self):
        """Test valid data upload request"""
        data = {
            "dataset_name": "Test Dataset",
            "description": "A test dataset",
            "target_column": "target"
        }
        
        model = DataUploadRequest(**data)
        assert model.dataset_name == "Test Dataset"
        assert model.description == "A test dataset"
        assert model.target_column == "target"
    
    def test_data_upload_request_empty_name(self):
        """Test empty dataset name"""
        data = {
            "dataset_name": "",
            "description": "A test dataset"
        }
        
        with pytest.raises(ValueError) as exc_info:
            DataUploadRequest(**data)
        
        assert "Dataset name cannot be empty" in str(exc_info.value)
#!/usr/bin/env python3
"""
Input Validation System

Provides comprehensive input validation for user interactions,
API inputs, and system configurations to improve user experience
and prevent errors.

Author: Claude Code
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import config

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation for the CPUC RAG system."""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def clear_messages(self):
        """Clear validation messages."""
        self.validation_errors.clear()
        self.validation_warnings.clear()
    
    def validate_proceeding_id(self, proceeding_id: str) -> Tuple[bool, str]:
        """
        Validate proceeding ID format and availability.
        
        Args:
            proceeding_id: The proceeding ID to validate
            
        Returns:
            Tuple of (is_valid, normalized_proceeding_id)
        """
        if not proceeding_id:
            self.validation_errors.append("Proceeding ID cannot be empty")
            return False, ""
        
        # Normalize input
        proceeding_id = proceeding_id.strip().upper()
        
        # Check format - should be like R2207005 or R.22-07-005
        if proceeding_id.startswith('R.'):
            # Convert R.22-07-005 to R2207005
            parts = proceeding_id[2:].split('-')
            if len(parts) == 3:
                try:
                    proceeding_id = f"R{parts[0]}{parts[1]}{parts[2]}"
                except:
                    self.validation_errors.append("Invalid proceeding format. Expected R.XX-XX-XXX or RXXXXXXX")
                    return False, ""
        
        # Validate standard format RXXXXXXX
        if not re.match(r'^R\d{7}$', proceeding_id):
            self.validation_errors.append("Proceeding ID must be in format RXXXXXXX (e.g., R2207005)")
            return False, ""
        
        # Check if proceeding is in configured list
        if proceeding_id not in config.SCRAPER_PROCEEDINGS:
            available_proceedings = ", ".join(config.SCRAPER_PROCEEDINGS[:5])
            if len(config.SCRAPER_PROCEEDINGS) > 5:
                available_proceedings += f" ... and {len(config.SCRAPER_PROCEEDINGS) - 5} more"
            
            self.validation_warnings.append(
                f"Proceeding {proceeding_id} not in configured list. "
                f"Available: {available_proceedings}"
            )
        
        return True, proceeding_id
    
    def validate_query_input(self, query: str) -> Tuple[bool, str]:
        """
        Validate user query input.
        
        Args:
            query: The user query to validate
            
        Returns:
            Tuple of (is_valid, cleaned_query)
        """
        if not query:
            self.validation_errors.append("Query cannot be empty")
            return False, ""
        
        # Clean and normalize query
        query = query.strip()
        
        # Length validation
        if len(query) < config.MIN_QUERY_LENGTH:
            self.validation_errors.append(
                f"Query too short. Minimum {config.MIN_QUERY_LENGTH} characters required"
            )
            return False, ""
        
        if len(query) > config.MAX_QUERY_LENGTH:
            self.validation_errors.append(
                f"Query too long. Maximum {config.MAX_QUERY_LENGTH} characters allowed"
            )
            return False, ""
        
        # Content validation
        if query.lower() in ['test', 'hello', 'hi', '?', '']:
            self.validation_warnings.append(
                "Query appears to be a test. For better results, try a specific question about CPUC proceedings"
            )
        
        # Check for potentially problematic content
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'onload=',
            r'onerror='
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.validation_errors.append("Query contains potentially unsafe content")
                return False, ""
        
        return True, query
    
    def validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL format and accessibility.
        
        Args:
            url: The URL to validate
            
        Returns:
            Tuple of (is_valid, normalized_url)
        """
        if not url:
            self.validation_errors.append("URL cannot be empty")
            return False, ""
        
        url = url.strip()
        
        # Basic URL format validation
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                # Try adding https://
                url = f"https://{url}"
                parsed = urlparse(url)
            
            if not parsed.scheme or not parsed.netloc:
                self.validation_errors.append("Invalid URL format")
                return False, ""
            
            # Check for supported schemes
            if parsed.scheme.lower() not in ['http', 'https']:
                self.validation_errors.append("Only HTTP and HTTPS URLs are supported")
                return False, ""
            
            # Check for CPUC domain preference
            if 'docs.cpuc.ca.gov' not in parsed.netloc.lower():
                self.validation_warnings.append(
                    "URL is not from docs.cpuc.ca.gov. Processing may be less reliable"
                )
            
            # Check for PDF extension
            if not url.lower().endswith('.pdf') and '/pdf' not in url.lower():
                self.validation_warnings.append(
                    "URL does not appear to be a PDF. Document processing may fail"
                )
            
            return True, url
            
        except Exception as e:
            self.validation_errors.append(f"URL validation failed: {str(e)}")
            return False, ""
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Tuple[bool, Path]:
        """
        Validate file path for existence and permissions.
        
        Args:
            file_path: The file path to validate
            
        Returns:
            Tuple of (is_valid, normalized_path)
        """
        if not file_path:
            self.validation_errors.append("File path cannot be empty")
            return False, Path()
        
        try:
            path = Path(file_path).resolve()
            
            # Check if path exists
            if not path.exists():
                self.validation_errors.append(f"File does not exist: {path}")
                return False, path
            
            # Check if it's a file
            if not path.is_file():
                self.validation_errors.append(f"Path is not a file: {path}")
                return False, path
            
            # Check read permissions
            if not path.stat().st_size > 0:
                self.validation_warnings.append(f"File appears to be empty: {path}")
            
            # Check file size (warn if very large)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:
                self.validation_warnings.append(
                    f"Large file detected ({file_size_mb:.1f}MB). Processing may take longer"
                )
            
            return True, path
            
        except Exception as e:
            self.validation_errors.append(f"File path validation failed: {str(e)}")
            return False, Path()
    
    def validate_batch_size(self, batch_size: Union[str, int]) -> Tuple[bool, int]:
        """
        Validate batch size parameter.
        
        Args:
            batch_size: The batch size to validate
            
        Returns:
            Tuple of (is_valid, validated_batch_size)
        """
        try:
            batch_size = int(batch_size)
        except (ValueError, TypeError):
            self.validation_errors.append("Batch size must be a valid integer")
            return False, 0
        
        if batch_size < 1:
            self.validation_errors.append("Batch size must be at least 1")
            return False, 0
        
        if batch_size > 1000:
            self.validation_errors.append("Batch size too large (maximum 1000)")
            return False, 0
        
        # Optimize based on available resources
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if batch_size > 64 and gpu_memory_gb < 8:
                    self.validation_warnings.append(
                        f"Large batch size ({batch_size}) may cause GPU memory issues. "
                        f"Consider using {min(batch_size, 32)} or less"
                    )
                elif batch_size > 128 and gpu_memory_gb < 12:
                    self.validation_warnings.append(
                        f"Very large batch size ({batch_size}) may cause GPU memory issues"
                    )
        except ImportError:
            pass
        
        return True, batch_size
    
    def validate_config_setting(self, key: str, value: Any) -> Tuple[bool, Any]:
        """
        Validate configuration setting.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            Tuple of (is_valid, validated_value)
        """
        # Define validation rules for specific config keys
        validation_rules = {
            'CHUNK_SIZE': {'type': int, 'min': 100, 'max': 5000},
            'CHUNK_OVERLAP': {'type': int, 'min': 0, 'max': 1000},
            'TOP_K_DOCS': {'type': int, 'min': 1, 'max': 100},
            'EMBEDDING_BATCH_SIZE': {'type': int, 'min': 1, 'max': 256},
            'VECTOR_STORE_BATCH_SIZE': {'type': int, 'min': 1, 'max': 2048},
            'SCRAPER_MAX_WORKERS': {'type': int, 'min': 1, 'max': 16},
            'URL_PARALLEL_WORKERS': {'type': int, 'min': 1, 'max': 8},
            'MAX_RETRY_ATTEMPTS': {'type': int, 'min': 0, 'max': 10},
            'PROCESSING_TIMEOUT_SECONDS': {'type': int, 'min': 30, 'max': 3600}
        }
        
        if key not in validation_rules:
            self.validation_warnings.append(f"No validation rule for config key: {key}")
            return True, value
        
        rule = validation_rules[key]
        
        # Type validation
        try:
            if rule['type'] == int:
                value = int(value)
            elif rule['type'] == float:
                value = float(value)
            elif rule['type'] == bool:
                if isinstance(value, str):
                    value = value.lower() in ['true', '1', 'yes', 'on']
                else:
                    value = bool(value)
        except (ValueError, TypeError):
            self.validation_errors.append(f"Invalid type for {key}. Expected {rule['type'].__name__}")
            return False, value
        
        # Range validation
        if 'min' in rule and value < rule['min']:
            self.validation_errors.append(f"{key} too small. Minimum: {rule['min']}")
            return False, value
        
        if 'max' in rule and value > rule['max']:
            self.validation_errors.append(f"{key} too large. Maximum: {rule['max']}")
            return False, value
        
        return True, value
    
    def validate_system_requirements(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate system requirements and resources.
        
        Returns:
            Tuple of (all_requirements_met, requirements_status)
        """
        requirements_status = {
            'python_version': True,
            'required_packages': True,
            'disk_space': True,
            'memory': True,
            'gpu_optional': True
        }
        
        # Python version check
        import sys
        if sys.version_info < (3, 8):
            requirements_status['python_version'] = False
            self.validation_errors.append("Python 3.8 or higher required")
        
        # Package availability check
        required_packages = config.REQUIRED_PACKAGES
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            requirements_status['required_packages'] = False
            self.validation_errors.append(f"Missing packages: {', '.join(missing_packages)}")
        
        # Disk space check
        try:
            import shutil
            total, used, free = shutil.disk_usage(config.PROJECT_ROOT)
            free_gb = free / (1024**3)
            
            if free_gb < 2:
                requirements_status['disk_space'] = False
                self.validation_errors.append(f"Insufficient disk space: {free_gb:.1f}GB available (minimum 2GB)")
            elif free_gb < 5:
                self.validation_warnings.append(f"Low disk space: {free_gb:.1f}GB available")
        except Exception as e:
            self.validation_warnings.append(f"Could not check disk space: {e}")
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 2:
                requirements_status['memory'] = False
                self.validation_errors.append(f"Insufficient memory: {available_gb:.1f}GB available (minimum 2GB)")
            elif available_gb < 4:
                self.validation_warnings.append(f"Low memory: {available_gb:.1f}GB available")
        except ImportError:
            self.validation_warnings.append("psutil not available for memory checking")
        
        # GPU check (optional)
        try:
            import torch
            if not torch.cuda.is_available():
                self.validation_warnings.append("GPU not available - will use CPU (slower processing)")
            else:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb < 4:
                    self.validation_warnings.append(f"Low GPU memory: {gpu_memory_gb:.1f}GB")
        except ImportError:
            self.validation_warnings.append("PyTorch not available for GPU checking")
        
        all_requirements_met = all(requirements_status.values())
        return all_requirements_met, requirements_status
    
    def get_validation_report(self) -> Dict[str, List[str]]:
        """Get validation report with errors and warnings."""
        return {
            'errors': self.validation_errors.copy(),
            'warnings': self.validation_warnings.copy()
        }
    
    def print_validation_report(self, title: str = "Validation Report"):
        """Print formatted validation report."""
        if not self.validation_errors and not self.validation_warnings:
            print(f"✅ {title}: All validations passed")
            return
        
        print(f"\n📋 {title}")
        print("=" * 50)
        
        if self.validation_errors:
            print(f"❌ Errors ({len(self.validation_errors)}):")
            for i, error in enumerate(self.validation_errors, 1):
                print(f"   {i}. {error}")
        
        if self.validation_warnings:
            print(f"⚠️  Warnings ({len(self.validation_warnings)}):")
            for i, warning in enumerate(self.validation_warnings, 1):
                print(f"   {i}. {warning}")


# Convenience functions for common validations

def validate_proceeding_input(proceeding_input: str) -> Tuple[bool, str, List[str], List[str]]:
    """
    Validate proceeding input and return normalized result.
    
    Returns:
        Tuple of (is_valid, normalized_proceeding, errors, warnings)
    """
    validator = InputValidator()
    is_valid, normalized = validator.validate_proceeding_id(proceeding_input)
    report = validator.get_validation_report()
    return is_valid, normalized, report['errors'], report['warnings']


def validate_query_input(query: str) -> Tuple[bool, str, List[str], List[str]]:
    """
    Validate query input and return cleaned result.
    
    Returns:
        Tuple of (is_valid, cleaned_query, errors, warnings)
    """
    validator = InputValidator()
    is_valid, cleaned = validator.validate_query_input(query)
    report = validator.get_validation_report()
    return is_valid, cleaned, report['errors'], report['warnings']


def validate_system_ready() -> Tuple[bool, Dict[str, Any], List[str], List[str]]:
    """
    Validate system readiness for operation.
    
    Returns:
        Tuple of (is_ready, status_dict, errors, warnings)
    """
    validator = InputValidator()
    is_ready, status = validator.validate_system_requirements()
    report = validator.get_validation_report()
    return is_ready, status, report['errors'], report['warnings']


if __name__ == "__main__":
    # Test validation functions
    validator = InputValidator()
    
    print("🧪 Testing Input Validation System")
    print("=" * 40)
    
    # Test proceeding validation
    test_proceedings = ["R2207005", "r2207005", "R.22-07-005", "invalid", ""]
    for proc in test_proceedings:
        validator.clear_messages()
        is_valid, normalized = validator.validate_proceeding_id(proc)
        print(f"Proceeding '{proc}': {'✅' if is_valid else '❌'} -> '{normalized}'")
        if not is_valid:
            validator.print_validation_report("Proceeding Validation")
    
    # Test system requirements
    print("\n🔍 System Requirements Check:")
    validator.clear_messages()
    is_ready, status = validator.validate_system_requirements()
    validator.print_validation_report("System Requirements")
    
    print(f"\n🎯 System Ready: {'✅' if is_ready else '❌'}")
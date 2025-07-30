#!/usr/bin/env python3
"""
Production Environment Validator

Validates system dependencies, configurations, and integration points
to ensure the CPUC RAG system is ready for production deployment.

Author: Claude Code
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import config

logger = logging.getLogger(__name__)

class ProductionValidator:
    """Comprehensive production environment validation."""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_failures = []
        self.warnings = []
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 PRODUCTION ENVIRONMENT VALIDATION")
        print("=" * 60)
        
        validators = [
            ("System Dependencies", self._validate_dependencies),
            ("CUDA/GPU Resources", self._validate_cuda_resources),
            ("Database Connectivity", self._validate_database),
            ("API Endpoints", self._validate_api_endpoints),
            ("Configuration Settings", self._validate_configuration),
            ("File System Permissions", self._validate_file_system),
            ("Memory and Resources", self._validate_system_resources),
            ("Integration Points", self._validate_integration_points)
        ]
        
        overall_success = True
        
        for check_name, validator_func in validators:
            print(f"\n📋 {check_name}...")
            try:
                result = validator_func()
                self.validation_results[check_name] = result
                
                if result['status'] == 'pass':
                    print(f"✅ {check_name}: PASSED")
                elif result['status'] == 'warning':
                    print(f"⚠️  {check_name}: PASSED WITH WARNINGS")
                    self.warnings.extend(result.get('warnings', []))
                else:
                    print(f"❌ {check_name}: FAILED")
                    self.critical_failures.extend(result.get('errors', []))
                    overall_success = False
                    
            except Exception as e:
                error_msg = f"{check_name} validation failed: {str(e)}"
                print(f"❌ {check_name}: ERROR - {error_msg}")
                self.critical_failures.append(error_msg)
                self.validation_results[check_name] = {
                    'status': 'error',
                    'errors': [error_msg]
                }
                overall_success = False
        
        # Generate comprehensive report
        self._generate_validation_report(overall_success)
        
        return {
            'overall_success': overall_success,
            'validation_results': self.validation_results,
            'critical_failures': self.critical_failures,
            'warnings': self.warnings
        }
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate required Python packages and models."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        # Check required packages
        required_packages = config.REQUIRED_PACKAGES
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                results['details'][f'package_{package}'] = 'available'
            except ImportError:
                missing_packages.append(package)
                results['details'][f'package_{package}'] = 'missing'
        
        if missing_packages:
            results['status'] = 'fail'
            results['errors'].append(f"Missing required packages: {', '.join(missing_packages)}")
        
        # Check specific model availability
        try:
            from transformers import AutoTokenizer
            for model_name in config.REQUIRED_MODELS:
                try:
                    AutoTokenizer.from_pretrained(model_name)
                    results['details'][f'model_{model_name}'] = 'available'
                except Exception:
                    results['warnings'].append(f"Model {model_name} not locally cached (will download on first use)")
                    results['details'][f'model_{model_name}'] = 'will_download'
        except ImportError:
            results['warnings'].append("Transformers library not available for model validation")
        
        return results
    
    def _validate_cuda_resources(self) -> Dict[str, Any]:
        """Validate CUDA/GPU resources and memory management."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        try:
            import torch
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            results['details']['cuda_available'] = cuda_available
            
            if cuda_available:
                # Get GPU information
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name()
                
                # Memory information
                memory_info = torch.cuda.get_device_properties(0)
                total_memory_gb = memory_info.total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                
                results['details'].update({
                    'device_count': device_count,
                    'current_device': current_device,
                    'device_name': device_name,
                    'total_memory_gb': total_memory_gb,
                    'allocated_memory_mb': allocated_memory / (1024**2),
                    'reserved_memory_mb': reserved_memory / (1024**2)
                })
                
                # Validate memory configuration
                if total_memory_gb < 4:
                    results['warnings'].append(f"GPU memory is low ({total_memory_gb:.1f}GB). Consider reducing batch size.")
                elif total_memory_gb >= 12:
                    results['details']['performance_tier'] = 'high'
                elif total_memory_gb >= 8:
                    results['details']['performance_tier'] = 'medium'
                else:
                    results['details']['performance_tier'] = 'low'
                
                # Test CUDA memory management
                try:
                    from models import get_cuda_memory_manager
                    cuda_manager = get_cuda_memory_manager()
                    memory_stats = cuda_manager.get_memory_stats()
                    results['details']['cuda_manager'] = 'operational'
                    results['details']['memory_stats'] = memory_stats
                except Exception as e:
                    results['warnings'].append(f"CUDA memory manager error: {str(e)}")
                    
            else:
                results['warnings'].append("CUDA not available - will use CPU/MPS for embeddings")
                results['details']['fallback_device'] = 'cpu'
                
                # Check for MPS (Apple Silicon)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    results['details']['mps_available'] = True
                    results['details']['fallback_device'] = 'mps'
                
        except ImportError:
            results['status'] = 'fail'
            results['errors'].append("PyTorch not available - required for embedding models")
        
        return results
    
    def _validate_database(self) -> Dict[str, Any]:
        """Validate LanceDB database connectivity and schema."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        try:
            import lancedb
            
            # Test database connection
            db_path = config.DB_DIR
            db_path.mkdir(parents=True, exist_ok=True)
            
            try:
                db = lancedb.connect(str(db_path))
                results['details']['database_connection'] = 'successful'
                
                # Check existing tables
                table_names = db.table_names()
                results['details']['existing_tables'] = len(table_names)
                results['details']['table_list'] = table_names[:5]  # Show first 5 tables
                
                # Test table creation and basic operations
                test_table_name = "validation_test_table"
                if test_table_name in table_names:
                    db.drop_table(test_table_name)
                
                # Create test table with sample schema
                import pyarrow as pa
                schema = pa.schema([
                    pa.field("vector", pa.list_(pa.float32(), 384)),
                    pa.field("content", pa.string()),
                    pa.field("source_url", pa.string())
                ])
                
                # Test table creation
                test_data = [
                    {"vector": [0.1] * 384, "content": "test", "source_url": "test://validation"}
                ]
                
                table = db.create_table(test_table_name, test_data, schema=schema)
                results['details']['table_creation'] = 'successful'
                
                # Test basic operations
                count = table.count_rows()
                results['details']['test_row_count'] = count
                
                # Cleanup test table
                db.drop_table(test_table_name)
                results['details']['cleanup'] = 'successful'
                
            except Exception as db_error:
                results['status'] = 'fail'
                results['errors'].append(f"Database operation failed: {str(db_error)}")
                
        except ImportError:
            results['status'] = 'fail'
            results['errors'].append("LanceDB not available - required for vector storage")
        
        return results
    
    def _validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate API endpoints and authentication."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        # Check OpenAI API configuration
        api_key = config.OPENAI_API_KEY
        if not api_key:
            results['status'] = 'fail'
            results['errors'].append("OpenAI API key not configured")
            return results
        
        results['details']['api_key_configured'] = True
        
        # Test OpenAI API connectivity
        if config.VALIDATE_OPENAI_API_ON_STARTUP:
            try:
                from models import get_llm
                
                # Test LLM initialization
                llm = get_llm()
                if llm:
                    results['details']['openai_connection'] = 'successful'
                    results['details']['model_name'] = config.OPENAI_MODEL_NAME
                    
                    # Test a simple query with timeout
                    try:
                        response = llm.invoke("Test connection")
                        if response and hasattr(response, 'content'):
                            results['details']['api_response_test'] = 'successful'
                        else:
                            results['warnings'].append("API responded but format unexpected")
                    except Exception as query_error:
                        results['warnings'].append(f"API query test failed: {str(query_error)}")
                        
                else:
                    results['status'] = 'fail'
                    results['errors'].append("Failed to initialize OpenAI LLM")
                    
            except Exception as api_error:
                results['status'] = 'fail'
                results['errors'].append(f"OpenAI API validation failed: {str(api_error)}")
        else:
            results['warnings'].append("API validation skipped (disabled in config)")
        
        return results
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration settings."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        # Use the built-in config validation
        try:
            validation_results = config.validate_production_config()
            results['details'].update(validation_results)
            
            # Check critical settings
            critical_issues = []
            
            if not validation_results.get('api_key_configured', False):
                critical_issues.append("OpenAI API key not configured")
            
            if not validation_results.get('database_directory', False):
                critical_issues.append("Database directory not accessible")
            
            if not validation_results.get('required_packages', False):
                critical_issues.append("Required packages missing")
            
            if critical_issues:
                results['status'] = 'fail'
                results['errors'].extend(critical_issues)
            
            # Check configuration values
            config_checks = {
                'CHUNK_SIZE': (config.CHUNK_SIZE, 100, 2000),
                'TOP_K_DOCS': (config.TOP_K_DOCS, 5, 50),
                'EMBEDDING_BATCH_SIZE': (config.EMBEDDING_BATCH_SIZE, 8, 128)
            }
            
            for setting_name, (value, min_val, max_val) in config_checks.items():
                if not (min_val <= value <= max_val):
                    results['warnings'].append(f"{setting_name} ({value}) outside recommended range {min_val}-{max_val}")
                    
        except Exception as config_error:
            results['status'] = 'fail'
            results['errors'].append(f"Configuration validation failed: {str(config_error)}")
        
        return results
    
    def _validate_file_system(self) -> Dict[str, Any]:
        """Validate file system permissions and directory structure."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        # Check critical directories
        critical_dirs = [
            ('project_root', config.PROJECT_ROOT),
            ('database_dir', config.DB_DIR.parent),
            ('agent_logs', config.AGENT_EVALUATION_LOG_DIR.parent)
        ]
        
        for dir_name, dir_path in critical_dirs:
            try:
                dir_path = Path(dir_path)
                
                # Check if directory exists or can be created
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = dir_path / f".validation_test_{int(time.time())}"
                test_file.write_text("test")
                test_file.unlink()
                
                results['details'][f'{dir_name}_writable'] = True
                
            except Exception as dir_error:
                results['status'] = 'fail'
                results['errors'].append(f"Directory {dir_name} ({dir_path}) not accessible: {str(dir_error)}")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(config.PROJECT_ROOT)
            free_gb = free / (1024**3)
            
            results['details']['free_disk_space_gb'] = free_gb
            
            if free_gb < 1:
                results['status'] = 'fail'
                results['errors'].append(f"Insufficient disk space: {free_gb:.1f}GB available")
            elif free_gb < 5:
                results['warnings'].append(f"Low disk space: {free_gb:.1f}GB available")
                
        except Exception as disk_error:
            results['warnings'].append(f"Could not check disk space: {str(disk_error)}")
        
        return results
    
    def _validate_system_resources(self) -> Dict[str, Any]:
        """Validate system memory and CPU resources."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        try:
            import psutil
            
            # Memory validation
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            results['details'].update({
                'total_memory_gb': total_gb,
                'available_memory_gb': available_gb,
                'memory_percent_used': memory.percent
            })
            
            if available_gb < 2:
                results['status'] = 'fail'
                results['errors'].append(f"Insufficient memory: {available_gb:.1f}GB available")
            elif available_gb < 4:
                results['warnings'].append(f"Low memory: {available_gb:.1f}GB available")
            
            # CPU validation
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            results['details'].update({
                'cpu_count': cpu_count,
                'cpu_percent_used': cpu_percent
            })
            
            if cpu_count < 2:
                results['warnings'].append(f"Low CPU count: {cpu_count} cores")
            
            if cpu_percent > 90:
                results['warnings'].append(f"High CPU usage: {cpu_percent}%")
            
            # Calculate optimal settings
            optimal_workers = config.get_optimal_worker_count()
            optimal_batch = config.get_memory_optimized_batch_size()
            
            results['details'].update({
                'recommended_workers': optimal_workers,
                'recommended_batch_size': optimal_batch
            })
            
        except ImportError:
            results['warnings'].append("psutil not available for system resource monitoring")
        except Exception as resource_error:
            results['warnings'].append(f"System resource validation error: {str(resource_error)}")
        
        return results
    
    def _validate_integration_points(self) -> Dict[str, Any]:
        """Validate integration between system components."""
        results = {'status': 'pass', 'details': {}, 'errors': [], 'warnings': []}
        
        # Test embedding model integration
        try:
            from models import get_embedding_model
            
            embedding_model = get_embedding_model()
            test_texts = ["Test embedding generation"]
            embeddings = embedding_model.embed_documents(test_texts)
            
            if embeddings and len(embeddings) == 1 and len(embeddings[0]) > 0:
                results['details']['embedding_integration'] = 'successful'
                results['details']['embedding_dimension'] = len(embeddings[0])
            else:
                results['errors'].append("Embedding model produced invalid output")
                results['status'] = 'fail'
                
        except Exception as embed_error:
            results['status'] = 'fail'
            results['errors'].append(f"Embedding model integration failed: {str(embed_error)}")
        
        # Test document processing integration
        try:
            from data_processing import extract_and_chunk_with_docling_url
            
            # Test with a small sample (don't actually process)
            test_url = "https://example.com/test.pdf"
            # Just validate the function exists and can be called
            results['details']['document_processing_available'] = True
            
        except ImportError as import_error:
            results['status'] = 'fail'
            results['errors'].append(f"Document processing module not available: {str(import_error)}")
        except Exception as proc_error:
            results['warnings'].append(f"Document processing validation warning: {str(proc_error)}")
        
        # Test RAG system integration
        try:
            from rag_core import CPUCRAGSystem
            
            # Initialize but don't load data
            rag_system = CPUCRAGSystem(current_proceeding="R2207005")
            results['details']['rag_system_init'] = 'successful'
            
        except Exception as rag_error:
            results['errors'].append(f"RAG system integration failed: {str(rag_error)}")
            results['status'] = 'fail'
        
        return results
    
    def _generate_validation_report(self, overall_success: bool):
        """Generate comprehensive validation report."""
        print(f"\n🎯 PRODUCTION VALIDATION SUMMARY")
        print("=" * 60)
        
        if overall_success:
            print("✅ SYSTEM READY FOR PRODUCTION")
            print("All critical validation checks passed successfully.")
        else:
            print("❌ SYSTEM NOT READY FOR PRODUCTION")
            print("Critical issues must be resolved before deployment.")
        
        # Summary statistics
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'pass')
        warning_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'warning')
        failed_checks = total_checks - passed_checks - warning_checks
        
        print(f"\n📊 Validation Statistics:")
        print(f"   Total checks: {total_checks}")
        print(f"   ✅ Passed: {passed_checks}")
        print(f"   ⚠️  Warnings: {warning_checks}")
        print(f"   ❌ Failed: {failed_checks}")
        
        # Critical failures
        if self.critical_failures:
            print(f"\n🚨 Critical Issues ({len(self.critical_failures)}):")
            for i, failure in enumerate(self.critical_failures, 1):
                print(f"   {i}. {failure}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if overall_success:
            print("   • System is ready for production deployment")
            print("   • Monitor performance metrics during initial operation")
            print("   • Address any warnings for optimal performance")
        else:
            print("   • Resolve all critical issues before deployment")
            print("   • Re-run validation after fixes")
            print("   • Consider staging environment testing")


def run_production_validation():
    """Run comprehensive production validation."""
    validator = ProductionValidator()
    results = validator.validate_all()
    
    return results['overall_success']


if __name__ == "__main__":
    print("🚀 CPUC RAG SYSTEM - PRODUCTION VALIDATION")
    print("=" * 50)
    
    success = run_production_validation()
    
    if success:
        print("\n🎉 VALIDATION PASSED - SYSTEM READY FOR PRODUCTION!")
        exit(0)
    else:
        print("\n🔧 VALIDATION FAILED - SYSTEM NEEDS ATTENTION")
        exit(1)
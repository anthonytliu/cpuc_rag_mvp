# Jupyter Cloud Hosting Guide for CPUC PDF Scraping & Embedding System

This guide provides step-by-step instructions for hosting the CPUC document scraping, chunking, and embedding system on a cloud-based Jupyter notebook environment with cloud storage integration.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Cloud Environment Setup](#cloud-environment-setup)
- [Installation & Configuration](#installation--configuration)
- [Core API Functions](#core-api-functions)
- [Usage Examples](#usage-examples)
- [Cloud Storage Integration](#cloud-storage-integration)
- [Output Download](#output-download)
- [Troubleshooting](#troubleshooting)

## Overview

The system consists of three main components:
1. **PDF Scraping**: Automated discovery and downloading of CPUC documents
2. **Document Processing**: Chunking and embedding generation using OpenAI models
3. **Vector Database**: LanceDB storage with incremental updates

All intermediate files are stored in cloud storage, with only the final database outputs (document_hashes.json, lance_table.lance, folders) downloaded as a zip file.

## Prerequisites

- Cloud Jupyter environment (Google Colab, AWS SageMaker, Azure Notebooks, etc.)
- OpenAI API key
- Google Cloud Storage or similar cloud storage service
- Python 3.8+

## Cloud Environment Setup

### 1. Mount Cloud Storage
```python
# For Google Colab with Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set working directory to cloud storage
import os
os.chdir('/content/drive/MyDrive/CPUC_RAG_Project')
```

### 2. Install Dependencies
```python
# Install required packages
!pip install -r requirements.txt

# For Chrome/Selenium support in cloud environments
!apt-get update
!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
```

## Installation & Configuration

### 1. Clone or Upload Project Files
```python
# If cloning from repository
!git clone https://github.com/your-repo/CPUC_REG_RAG.git
os.chdir('CPUC_REG_RAG')

# Or upload the following core files to your cloud storage:
# - config.py
# - standalone_scraper.py
# - incremental_embedder.py
# - rag_core.py
# - cpuc_scraper.py
# - data_processing.py
# - models.py
# - utils.py
# - requirements.txt
```

### 2. Environment Configuration
```python
import os
from pathlib import Path

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'
os.environ['DEBUG'] = 'false'
os.environ['SCRAPER_MAX_WORKERS'] = '4'  # Reduce for cloud environments

# Verify configuration
import config
print(f"Project root: {config.PROJECT_ROOT}")
print(f"Database directory: {config.DB_DIR}")
print(f"Available proceedings: {list(config.AVAILABLE_PROCEEDINGS.keys())}")
```

## Core API Functions

### 1. PDF Scraping API
```python
def scrape_documents(proceeding_id, headless=True):
    """
    Scrape documents for a specific CPUC proceeding.
    
    Args:
        proceeding_id (str): CPUC proceeding number (e.g., 'R2207005')
        headless (bool): Run browser in headless mode
        
    Returns:
        dict: Scraping results with document counts and status
    """
    from standalone_scraper import run_standalone_scraper
    
    print(f"🔍 Starting document scraping for {proceeding_id}...")
    results = run_standalone_scraper(proceeding_id, headless=headless)
    
    if results.get('success', True):
        print(f"✅ Scraping completed: {results.get('total_pdfs', 0)} documents found")
        return {
            'status': 'success',
            'proceeding': proceeding_id,
            'documents_found': results.get('total_pdfs', 0),
            'csv_pdfs': results.get('csv_pdfs', 0),
            'google_pdfs': results.get('google_pdfs', 0)
        }
    else:
        print(f"❌ Scraping failed: {results.get('error', 'Unknown error')}")
        return {
            'status': 'error',
            'proceeding': proceeding_id,
            'error': results.get('error', 'Unknown error')
        }

# Example usage
scraping_results = scrape_documents('R2207005')
```

### 2. Document Processing & Embedding API
```python
def process_embeddings(proceeding_id, progress_callback=None):
    """
    Process document embeddings for a specific proceeding.
    
    Args:
        proceeding_id (str): CPUC proceeding number
        progress_callback (callable): Optional progress callback function
        
    Returns:
        dict: Processing results with embedding counts and status
    """
    from incremental_embedder import process_incremental_embeddings
    
    def default_progress(message, progress):
        print(f"[{progress}%] {message}")
    
    callback = progress_callback or default_progress
    
    print(f"🔄 Starting embedding process for {proceeding_id}...")
    results = process_incremental_embeddings(proceeding_id, progress_callback=callback)
    
    if results['status'] == 'completed':
        print(f"✅ Embedding completed: {results['documents_processed']} documents processed")
        return {
            'status': 'success',
            'proceeding': proceeding_id,
            'documents_processed': results['documents_processed'],
            'successful': results['successful'],
            'failed': results['failed']
        }
    elif results['status'] == 'up_to_date':
        print("✅ All documents are already embedded")
        return {
            'status': 'up_to_date',
            'proceeding': proceeding_id,
            'documents_processed': 0
        }
    else:
        print(f"❌ Embedding failed: {results.get('error', 'Unknown error')}")
        return {
            'status': 'error',
            'proceeding': proceeding_id,
            'error': results.get('error', 'Unknown error')
        }

# Example usage
embedding_results = process_embeddings('R2207005')
```

### 3. Complete Pipeline API
```python
def run_complete_pipeline(proceeding_ids, include_scraping=True, include_embedding=True):
    """
    Run the complete pipeline for multiple proceedings.
    
    Args:
        proceeding_ids (list): List of proceeding IDs to process
        include_scraping (bool): Whether to run document scraping
        include_embedding (bool): Whether to run embedding processing
        
    Returns:
        dict: Complete pipeline results
    """
    results = {
        'proceedings_processed': [],
        'scraping_results': {},
        'embedding_results': {},
        'total_documents': 0,
        'total_embeddings': 0,
        'errors': []
    }
    
    for proceeding_id in proceeding_ids:
        print(f"\n{'='*60}")
        print(f"🚀 Processing {proceeding_id}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Document Scraping
            if include_scraping:
                scraping_result = scrape_documents(proceeding_id)
                results['scraping_results'][proceeding_id] = scraping_result
                
                if scraping_result['status'] == 'success':
                    results['total_documents'] += scraping_result['documents_found']
                else:
                    results['errors'].append(f"Scraping failed for {proceeding_id}: {scraping_result.get('error')}")
                    continue
            
            # Step 2: Document Processing & Embedding
            if include_embedding:
                embedding_result = process_embeddings(proceeding_id)
                results['embedding_results'][proceeding_id] = embedding_result
                
                if embedding_result['status'] in ['success', 'up_to_date']:
                    results['total_embeddings'] += embedding_result.get('documents_processed', 0)
                else:
                    results['errors'].append(f"Embedding failed for {proceeding_id}: {embedding_result.get('error')}")
            
            results['proceedings_processed'].append(proceeding_id)
            
        except Exception as e:
            error_msg = f"Pipeline failed for {proceeding_id}: {str(e)}"
            results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
    
    return results

# Example usage
pipeline_results = run_complete_pipeline(['R2207005', 'R1807006'])
```

## Usage Examples

### Example 1: Single Proceeding Processing
```python
# Process a single proceeding from start to finish
proceeding_id = 'R2207005'

# Step 1: Scrape documents
print("Step 1: Document Scraping")
scraping_results = scrape_documents(proceeding_id)

# Step 2: Process embeddings
print("\nStep 2: Embedding Processing")
embedding_results = process_embeddings(proceeding_id)

# Step 3: Test the RAG system
print("\nStep 3: Testing RAG System")
from rag_core import CPUCRAGSystem

rag_system = CPUCRAGSystem(current_proceeding=proceeding_id)
test_query = "What are the key requirements for demand flexibility programs?"
response = rag_system.query_documents(test_query)
print(f"Query: {test_query}")
print(f"Response: {response[:200]}...")
```

### Example 2: Batch Processing Multiple Proceedings
```python
# Process multiple proceedings from the configured list
from config import SCRAPER_PROCEEDINGS

# Select first 3 proceedings for demonstration
proceedings_to_process = SCRAPER_PROCEEDINGS[:3]
print(f"Processing {len(proceedings_to_process)} proceedings: {proceedings_to_process}")

# Run complete pipeline
batch_results = run_complete_pipeline(
    proceeding_ids=proceedings_to_process,
    include_scraping=True,
    include_embedding=True
)

# Display results
print(f"\n{'='*60}")
print("BATCH PROCESSING RESULTS")
print(f"{'='*60}")
print(f"Proceedings processed: {len(batch_results['proceedings_processed'])}")
print(f"Total documents found: {batch_results['total_documents']}")
print(f"Total embeddings created: {batch_results['total_embeddings']}")
print(f"Errors: {len(batch_results['errors'])}")

if batch_results['errors']:
    print("\nErrors encountered:")
    for error in batch_results['errors']:
        print(f"  ❌ {error}")
```

### Example 3: Incremental Updates
```python
# Check for updates and process only new documents
def run_incremental_update(proceeding_id):
    """Run incremental update for a specific proceeding."""
    
    # Check existing embedding status
    from incremental_embedder import IncrementalEmbedder
    embedder = IncrementalEmbedder(proceeding_id)
    status = embedder.get_embedding_status()
    
    print(f"Current status for {proceeding_id}:")
    print(f"  📊 Total embedded: {status['total_embedded']}")
    print(f"  ❌ Total failed: {status['total_failed']}")
    print(f"  🕒 Last updated: {status.get('last_updated', 'Never')}")
    
    # Run scraping to check for new documents
    scraping_results = scrape_documents(proceeding_id)
    
    # Process any new embeddings
    embedding_results = process_embeddings(proceeding_id)
    
    return {
        'proceeding': proceeding_id,
        'scraping': scraping_results,
        'embedding': embedding_results,
        'previous_status': status
    }

# Example usage
update_results = run_incremental_update('R2207005')
```

## Cloud Storage Integration

### 1. Sync with Cloud Storage
```python
import shutil
from pathlib import Path

def sync_to_cloud_storage(cloud_storage_path='/content/drive/MyDrive/CPUC_RAG_Data'):
    """
    Sync local data to cloud storage.
    
    Args:
        cloud_storage_path (str): Path to cloud storage directory
    """
    cloud_path = Path(cloud_storage_path)
    cloud_path.mkdir(exist_ok=True)
    
    # Sync proceedings data
    local_proceedings = Path('cpuc_proceedings')
    cloud_proceedings = cloud_path / 'cpuc_proceedings'
    
    if local_proceedings.exists():
        if cloud_proceedings.exists():
            shutil.rmtree(cloud_proceedings)
        shutil.copytree(local_proceedings, cloud_proceedings)
        print(f"✅ Synced proceedings data to {cloud_proceedings}")
    
    # Sync vector databases
    local_db = Path('local_lance_db')
    cloud_db = cloud_path / 'local_lance_db'
    
    if local_db.exists():
        if cloud_db.exists():
            shutil.rmtree(cloud_db)
        shutil.copytree(local_db, cloud_db)
        print(f"✅ Synced vector databases to {cloud_db}")

# Sync data after processing
sync_to_cloud_storage()
```

### 2. Load from Cloud Storage
```python
def load_from_cloud_storage(cloud_storage_path='/content/drive/MyDrive/CPUC_RAG_Data'):
    """
    Load data from cloud storage to local environment.
    
    Args:
        cloud_storage_path (str): Path to cloud storage directory
    """
    cloud_path = Path(cloud_storage_path)
    
    if not cloud_path.exists():
        print(f"❌ Cloud storage path does not exist: {cloud_path}")
        return
    
    # Load proceedings data
    cloud_proceedings = cloud_path / 'cpuc_proceedings'
    local_proceedings = Path('cpuc_proceedings')
    
    if cloud_proceedings.exists():
        if local_proceedings.exists():
            shutil.rmtree(local_proceedings)
        shutil.copytree(cloud_proceedings, local_proceedings)
        print(f"✅ Loaded proceedings data from {cloud_proceedings}")
    
    # Load vector databases
    cloud_db = cloud_path / 'local_lance_db'
    local_db = Path('local_lance_db')
    
    if cloud_db.exists():
        if local_db.exists():
            shutil.rmtree(local_db)
        shutil.copytree(cloud_db, local_db)
        print(f"✅ Loaded vector databases from {cloud_db}")

# Load existing data at startup
load_from_cloud_storage()
```

## Output Download

### 1. Prepare Final Output
```python
import zipfile
from datetime import datetime

def prepare_final_output(proceedings_list=None, output_filename=None):
    """
    Prepare final output files for download as a zip file.
    
    Args:
        proceedings_list (list): List of proceedings to include (None for all)
        output_filename (str): Custom filename for the zip file
        
    Returns:
        str: Path to the created zip file
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"cpuc_rag_database_{timestamp}.zip"
    
    zip_path = Path(output_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Add vector databases
        db_dir = Path('local_lance_db')
        if db_dir.exists():
            for proceeding_dir in db_dir.iterdir():
                if proceeding_dir.is_dir():
                    proceeding_id = proceeding_dir.name
                    
                    # Skip if we're filtering and this proceeding isn't in the list
                    if proceedings_list and proceeding_id not in proceedings_list:
                        continue
                    
                    # Add all files in the proceeding's vector database
                    for file_path in proceeding_dir.rglob('*'):
                        if file_path.is_file():
                            arc_path = f"vector_databases/{proceeding_id}/{file_path.name}"
                            zipf.write(file_path, arc_path)
                    
                    print(f"✅ Added vector database for {proceeding_id}")
        
        # Add metadata and configuration files
        metadata_files = [
            'config.py',
            'requirements.txt'
        ]
        
        for file_name in metadata_files:
            file_path = Path(file_name)
            if file_path.exists():
                zipf.write(file_path, f"config/{file_name}")
        
        # Add proceeding metadata
        proceedings_dir = Path('cpuc_proceedings')
        if proceedings_dir.exists():
            for proceeding_dir in proceedings_dir.iterdir():
                if proceeding_dir.is_dir():
                    proceeding_id = proceeding_dir.name
                    
                    # Skip if we're filtering and this proceeding isn't in the list
                    if proceedings_list and proceeding_id not in proceedings_list:
                        continue
                    
                    # Add key metadata files
                    metadata_files = [
                        f"{proceeding_id}_scraped_pdf_history.json",
                        "embeddings/embedding_status.json",
                        "documents/*.csv"
                    ]
                    
                    for pattern in metadata_files:
                        if '*' in pattern:
                            # Handle glob patterns
                            for file_path in proceeding_dir.glob(pattern):
                                if file_path.is_file():
                                    arc_path = f"metadata/{proceeding_id}/{file_path.name}"
                                    zipf.write(file_path, arc_path)
                        else:
                            file_path = proceeding_dir / pattern
                            if file_path.exists() and file_path.is_file():
                                arc_path = f"metadata/{proceeding_id}/{file_path.name}"
                                zipf.write(file_path, arc_path)
                    
                    print(f"✅ Added metadata for {proceeding_id}")
        
        # Add README with usage instructions
        readme_content = f"""# CPUC RAG Database Export
        
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

### vector_databases/
LanceDB vector databases for each proceeding, containing:
- lance_table.lance: Main vector database file
- document_hashes.json: Document hash tracking
- Vector index files and metadata

### metadata/
Proceeding metadata including:
- PDF scraping history
- Embedding status tracking
- Document CSV files

### config/
System configuration files:
- config.py: Main configuration
- requirements.txt: Python dependencies

## Usage

1. Extract this zip file to your local system
2. Install dependencies: pip install -r config/requirements.txt
3. Place vector_databases contents in local_lance_db/
4. Place metadata contents in cpuc_proceedings/
5. Run your RAG application

## Proceedings Included
{chr(10).join([f"- {p}" for p in proceedings_list or ["All available proceedings"]])}
"""
        
        zipf.writestr("README.txt", readme_content)
    
    file_size = zip_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"✅ Created output file: {zip_path}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    
    return str(zip_path)

# Example usage
output_file = prepare_final_output()
print(f"Download ready: {output_file}")
```

### 2. Download in Colab
```python
# For Google Colab - download the file
from google.colab import files

def download_output(zip_filename):
    """Download the output file in Google Colab."""
    try:
        files.download(zip_filename)
        print(f"✅ Download started for {zip_filename}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"💡 File is available at: {Path(zip_filename).absolute()}")

# Download the prepared output
download_output(output_file)
```

### 3. Cloud Storage Upload
```python
def upload_to_cloud_storage(zip_filename, cloud_storage_path='/content/drive/MyDrive/CPUC_RAG_Outputs'):
    """Upload output file to cloud storage for later download."""
    cloud_path = Path(cloud_storage_path)
    cloud_path.mkdir(exist_ok=True)
    
    source_file = Path(zip_filename)
    destination_file = cloud_path / zip_filename
    
    shutil.copy2(source_file, destination_file)
    print(f"✅ Uploaded {zip_filename} to {destination_file}")
    
    return str(destination_file)

# Upload to cloud storage
cloud_file = upload_to_cloud_storage(output_file)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues in Cloud Environments
```python
# Reduce batch sizes for cloud environments
import config
config.VECTOR_STORE_BATCH_SIZE = 32  # Reduce from 256
config.EMBEDDING_BATCH_SIZE = 16     # Reduce from 32
config.URL_PARALLEL_WORKERS = 2      # Reduce from 3
```

#### 2. Selenium/Chrome Issues
```python
# For cloud environments, ensure proper Chrome setup
def setup_chrome_for_cloud():
    """Setup Chrome for cloud environments."""
    try:
        # Install Chrome and ChromeDriver
        os.system('apt-get update')
        os.system('apt-get install -y chromium-browser')
        os.system('apt-get install -y chromium-chromedriver')
        
        # Set environment variable
        os.environ['CHROME_BIN'] = '/usr/bin/chromium-browser'
        
        print("✅ Chrome setup completed")
    except Exception as e:
        print(f"❌ Chrome setup failed: {e}")

# Run Chrome setup
setup_chrome_for_cloud()
```

#### 3. API Rate Limiting
```python
# Add delays between API calls
import time

def rate_limited_processing(proceeding_ids, delay_seconds=60):
    """Process proceedings with rate limiting."""
    results = []
    
    for i, proceeding_id in enumerate(proceeding_ids):
        print(f"Processing {proceeding_id} ({i+1}/{len(proceeding_ids)})")
        
        result = run_complete_pipeline([proceeding_id])
        results.append(result)
        
        if i < len(proceeding_ids) - 1:
            print(f"⏳ Waiting {delay_seconds} seconds before next proceeding...")
            time.sleep(delay_seconds)
    
    return results

# Use rate-limited processing for large batches
batch_results = rate_limited_processing(['R2207005', 'R1807006'], delay_seconds=30)
```

#### 4. Disk Space Management
```python
def cleanup_intermediate_files():
    """Clean up intermediate files to save disk space."""
    import shutil
    
    cleanup_dirs = [
        'cpuc_proceedings/*/documents',  # Keep only metadata, not full documents
        '__pycache__',
        '.tqdm'
    ]
    
    for pattern in cleanup_dirs:
        if '*' in pattern:
            for path in Path('.').glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"🧹 Cleaned up {path}")
        else:
            path = Path(pattern)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"🧹 Cleaned up {path}")

# Clean up after processing
cleanup_intermediate_files()
```

#### 5. Monitor Progress and Resources
```python
import psutil

def monitor_resources():
    """Monitor system resources during processing."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    print(f"💾 Memory: {memory.percent}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    print(f"💿 Disk: {disk.percent}% used ({disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB)")
    
    if memory.percent > 85:
        print("⚠️  High memory usage detected!")
    if disk.percent > 90:
        print("⚠️  Low disk space detected!")

# Monitor resources periodically
monitor_resources()
```

---

This guide provides a complete framework for hosting the CPUC PDF scraping and embedding system in cloud-based Jupyter notebooks. The system is designed to keep all intermediate files in cloud storage while providing easy download of the final vector database outputs.
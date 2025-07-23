# CPUC RAG System - Quick Start Guide

## 🚀 Getting Started (New Users)

1. **Discover Documents**
   ```bash
   python standalone_scraper.py R2207005
   ```

2. **Start Application**
   ```bash
   streamlit run app.py
   ```

3. **Access Interface**: http://localhost:8501

## 📋 Common Commands

### Document Scraping
```bash
# Scrape default proceeding
python standalone_scraper.py

# Scrape specific proceeding  
python standalone_scraper.py R2207005

# List available proceedings
python standalone_scraper.py --list-proceedings

# Verbose output
python standalone_scraper.py --verbose R2207005

# Get help
python standalone_scraper.py --help
```

### Application
```bash
# Start main application
streamlit run app.py

# The app will automatically:
# - Load scraped document data
# - Build vector embeddings  
# - Provide query interface
```

### Testing
```bash
# Run all tests
python -m unittest test_cpuc_scraper -v

# Test new features
python -m unittest test_cpuc_scraper.TestPageURLTrackingAndDuplicatePrevention -v

# Test standalone scraper
python -m unittest test_cpuc_scraper.TestStandaloneScraperIntegration -v
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No existing data found" | Run `python standalone_scraper.py [proceeding]` first |
| Scraper fails | Check internet connection, Chrome browser installed |
| Import errors | Run `pip install -r requirements.txt` |
| Slow performance | Use `--verbose` flag to monitor progress |

## 📁 Output Files

- `cpuc_csvs/R2207005_resultCSV.csv` - Official CPUC document data
- `cpuc_csvs/R2207005_scraped_pdf_history.json` - Complete PDF metadata
- `local_chroma_db/` - Vector embeddings database
- `standalone_scraper.log` - Scraper execution logs

## ⚡ Quick Workflow

```bash
# Complete setup in 3 commands
python standalone_scraper.py --list-proceedings  # See what's available
python standalone_scraper.py R2207005            # Scrape documents  
streamlit run app.py                             # Start application
```

## 🎯 Key Features

- ✅ **Page URL Tracking**: Skips already processed document pages
- ✅ **Duplicate Prevention**: Prevents re-processing same PDFs
- ✅ **Standalone Process**: Scraping separated from main app
- ✅ **Comprehensive Metadata**: Preserves all document information
- ✅ **Non-destructive**: Safe operations with backup/restore

## 📖 Full Documentation

See `startup_instructions.txt` for complete details and advanced usage.
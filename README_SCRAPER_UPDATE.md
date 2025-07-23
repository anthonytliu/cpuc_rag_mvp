# CPUC RAG System - Scraper Architecture Update

## 🔄 Recent Changes

The CPUC document scraper has been **separated from the main application** for better performance and maintainability.

## 🚀 New Workflow

### Before (Old)
```bash
streamlit run app.py  # App would auto-scrape on startup
```

### Now (New)
```bash
# Step 1: Discover documents (standalone process)
python standalone_scraper.py R2207005

# Step 2: Start application  
streamlit run app.py
```

## ✨ Benefits

- **Performance**: No scraping delays during app startup
- **Reliability**: Scraping can be run independently and retried if needed
- **Control**: Manual control over when document discovery occurs
- **Efficiency**: Advanced duplicate prevention and page URL tracking

## 📋 Manual Scraper Commands

```bash
# Basic usage
python standalone_scraper.py R2207005

# See available proceedings
python standalone_scraper.py --list-proceedings

# Verbose logging
python standalone_scraper.py --verbose R2207005

# Help
python standalone_scraper.py --help
```

## 🔧 Integration Points

The standalone scraper creates these files that the main app uses:
- `cpuc_csvs/[proceeding]_resultCSV.csv`
- `cpuc_csvs/[proceeding]_scraped_pdf_history.json`

## 📖 Full Documentation

- **Complete Instructions**: `startup_instructions.txt`
- **Quick Reference**: `quick_start.md`
- **Test Coverage**: `test_cpuc_scraper.py`

## 🛠️ For Developers

The scraper implements these new features:
1. **Page URL Tracking**: Skips already processed document pages
2. **Google Search Deduplication**: Prevents duplicate URL processing
3. **Enhanced Metadata Preservation**: Saves complete PDF information
4. **Standalone Architecture**: Clean separation from main application
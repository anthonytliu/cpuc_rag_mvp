# Standalone Scraper Usage Guide

The `standalone_scraper.py` script allows you to scrape CPUC proceedings independently from the main RAG application.

## Quick Start

### Scrape All Proceedings (Default Behavior)
```bash
python standalone_scraper.py
```
**This will scrape ALL 37 proceedings listed in `config.SCRAPER_PROCEEDINGS` starting from the first entry (R2207005).**

### Scrape a Specific Proceeding
```bash
python standalone_scraper.py R2207005
```

### List Available Proceedings
```bash
python standalone_scraper.py --list-proceedings
```

### Get Help
```bash
python standalone_scraper.py --help
```

## Default Behavior Explained

When you run `python standalone_scraper.py` with **no arguments**, the script will:

1. **Load all proceedings** from `config.SCRAPER_PROCEEDINGS` (currently 37 proceedings)
2. **Process them sequentially** starting from the first entry in the list
3. **Create output files** for each proceeding in the `cpuc_proceedings/` directory
4. **Log progress** for each proceeding as it's processed

## Current Proceedings List (from config.py)

The script will process these proceedings in order:

1. R2207005 - Advance Demand Flexibility Through Electric Rates
2. R1807006 - Affordability Rulemaking  
3. R1901011 - Building Decarbonization
4. R1202009 - CCA Code of Conduct
5. R0310003 - CCA Rulemaking
6. ... (and 32 more proceedings)

## Output Files

For each proceeding, the scraper creates:
- **CSV file**: `cpuc_proceedings/{proceeding}/documents/{proceeding}_documents.csv`
- **History file**: `cpuc_proceedings/{proceeding}/{proceeding}_scraped_pdf_history.json`

## Example Output

```bash
$ python standalone_scraper.py

🔍 CPUC Standalone Scraper
==================================================
Proceedings: 37 total
  1. R2207005
  2. R1807006
  3. R1901011
  ...
  37. R1810007
Headless Mode: True
==================================================

📊 Processing proceeding 1/37: R2207005
✅ Completed R2207005: 156 PDFs
📊 Processing proceeding 2/37: R1807006  
✅ Completed R1807006: 234 PDFs
...

✅ Scraping completed!
📊 Summary: 35/37 successful
🗂️  Total PDFs: 8,429
📈 Success Rate: 94.6%
```

## CSV Age Optimization

The scraper now includes intelligent CSV age checking to improve efficiency:

### CSV File Age Logic
- **Recent CSV** (< 7 days old): Skips CSV download, only runs Google search for new PDFs
- **Old CSV** (≥ 7 days old): Downloads fresh CSV and runs full scraping process  
- **No CSV**: Runs full scraping process

### Example Behavior
```bash
# For a proceeding with recent CSV
📋 Found existing CSV file: cpuc_proceedings/R2207005/documents/R2207005_documents.csv
   Size: 1,234,567 bytes
   Age: 3 days old
   Status: Recent
⏭️  Skipping CSV scraping (file is 3 days old)
🔍 Running Google search only for R2207005...
✅ Google search completed successfully!
📊 Results Summary (Google Search Only):
   • Existing CSV PDFs: 156
   • New Google PDFs: 3
   • Total PDFs: 159
   • CSV file age: 3 days

# For a proceeding with old CSV
📋 Found existing CSV file: cpuc_proceedings/R1807006/documents/R1807006_documents.csv
   Size: 987,654 bytes  
   Age: 10 days old
   Status: Stale
🔄 CSV file is 10 days old - running full scraping
✅ Full scraping completed successfully!
```

## Performance Notes

- Processing all 37 proceedings is now much faster due to CSV age optimization
- Recent CSVs (< 7 days) only perform Google search, saving significant time
- The script runs in headless mode by default for better performance
- Each proceeding is processed sequentially to avoid overwhelming the CPUC website
- Progress is logged and can be monitored in real-time

## Integration with Data Processor

After running the scraper, you can process the discovered documents with:

```bash
python standalone_data_processor.py        # Process all proceedings
python standalone_data_processor.py R2207005  # Process specific proceeding
```
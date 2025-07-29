# CPUC RAG System: Complete Startup Flow Diagram

## Text-Based Flow Representation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CPUC RAG SYSTEM STARTUP SEQUENCE                     │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┐
    │   streamlit run     │ ◄── Entry Point
    │      app.py         │     
    └──────────┬──────────┘
               │ 
               ▼
    ┌─────────────────────┐
    │   Module Imports    │ ◄── app.py:11-21 
    │  Config, RAG, etc   │     Import dependencies
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Streamlit Init     │ ◄── app.py:26-51
    │ Page config & CSS   │     UI setup
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   main() Function   │ ◄── app.py:805
    │    Entry point      │     Main execution
    └──────────┬──────────┘
               │
               ▼
          ┌─────────────┐
          │  Startup    │ ◄── app.py:809
          │ Complete?   │     Session state check
          └──────┬──────┘
                 │
        ┌────────┴────────┐
        │ No             │ Yes
        ▼                ▼
┌──────────────────┐  ┌──────────────────┐
│ Execute Enhanced │  │  Skip Startup    │ ◄── Load existing
│ Startup Sequence │  │   Sequence       │     RAG system
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         ▼                     │
┌──────────────────┐           │
│ Create Startup   │ ◄── app.py:84 ──┘
│    Manager       │     startup_manager.py:380
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ StartupManager   │ ◄── startup_manager.py:32-44
│   Initialize     │     Progress callback setup
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Execute Startup  │ ◄── startup_manager.py:46
│   Sequence       │     0% Progress
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Select First     │ ◄── startup_manager.py:63-71
│   Proceeding     │     config.py:66-68, 10% Progress
└────────┬─────────┘     DEFAULT_PROCEEDING (R2207005)
         │
         ▼
┌──────────────────┐
│ Initialize DB    │ ◄── startup_manager.py:74-82
│   & Folders      │     Creates DB folders, 20% Progress
└────────┬─────────┘     Triggers RAG system init
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          RAG SYSTEM INITIALIZATION                              │
└──────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ RAG System Core  │ ◄── rag_core.py:34-72
│    Initialize    │     Embeddings & LLM setup
└────────┬─────────┘     models.get_embedding_model()
         │               models.get_llm()
         ▼
┌──────────────────┐
│ Vector Store     │ ◄── rag_core.py:70
│    Loading       │     _load_existing_vector_store()
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT DISCOVERY & PROCESSING                          │
└──────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ Scraper Workflow │ ◄── startup_manager.py:86-101
│   Execution      │     CPUCSimplifiedScraper
└────────┬─────────┘     30-65% Progress
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  CSV Data      │ │ Google Search  │ │ Update PDF     │ │ URL Processing │
│   Fetch        │ │ Additional     │ │   History      │ │   Pipeline     │
└────────────────┘ │ Documents      │ └────────────────┘ └────────────────┘
                   └────────────────┘
                   cpuc.ca.gov filter  scraped_pdf_history.json
                          │                       │
                          └───────────┬───────────┘
                                      ▼
                          ┌─────────────────────┐
                          │  Document URL       │ ◄── rag_core.py:354-517
                          │   Processing        │     build_vector_store_from_urls()
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ URL          │ │ Parallel     │ │ URL          │
            │ Validation   │ │ Processing   │ │ Hashing      │
            └──────────────┘ └──────────────┘ └──────────────┘
            data_processing.py  rag_core.py     data_processing.py
            :66-97             :460-499        :46-64
                    │                ▼                │
                    └────────────────┬────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │ Document Chunking   │ ◄── data_processing.py:120+
                          │   & Extraction      │     extract_and_chunk_with_docling_url()
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Docling      │ │ Content      │ │ Metadata     │
            │ Processing   │ │ Chunking     │ │ Enhancement  │
            └──────────────┘ └──────────────┘ └──────────────┘
            PDF→structured   Text, tables    Proceeding-specific
            content         metadata        superseding logic
                    │                ▼                │
                    └────────────────┬────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING PROCESSING                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ Incremental         │ ◄── startup_manager.py:104-123
                          │ Embedding           │     70-100% Progress
                          │ Processing          │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                 ▼
            ┌──────────────────────┐         ┌──────────────────────┐
            │ Primary: Incremental │         │ Fallback: Standard   │
            │ Embedder             │         │ RAG Build            │
            └──────────────────────┘         └──────────────────────┘
            incremental_embedder.py          startup_manager.py
            :57-100                          :297-341
            process_incremental_embeddings()  _run_standard_embedding()
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM FINALIZATION                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ Startup             │ ◄── startup_manager.py:124-143
                          │ Completion          │     Results compilation
                          └──────────┬──────────┘     Error/warning counts
                                     ▼
                          ┌─────────────────────┐
                          │ Session State       │ ◄── app.py:88-101
                          │ Population          │     RAG system storage
                          └──────────┬──────────┘     startup_completed = True
                                     ▼
                          ┌─────────────────────┐
                          │ Background Services │ ◄── app.py:915-923
                          │ Initialization      │     PDF scheduler
                          └──────────┬──────────┘     Background processor
                                     ▼
                          ┌─────────────────────┐
                          │ UI Interface        │ ◄── app.py:875-936
                          │ Ready               │     Main application interface
                          └─────────────────────┘     System ready for queries

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ERROR HANDLING                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

Error: Vector Corruption  ──────► Rebuild from scraped history
Error: Scraper Failure    ──────► Continue with existing data
Error: Embedder Unavail   ──────► Fallback to standard RAG build
Error: PDF Processing     ──────► Log errors, continue with remaining docs
```

## Key File System Structure Created:

```
local_lance_db/
├── {proceeding}/                    # Vector database storage
├── document_hashes.json             # URL hash tracking for deduplication
│
cpuc_csvs/
├── {proceeding}_scraped_pdf_history.json  # PDF metadata and status
├── {proceeding}_resultCSV.csv             # CPUC search results
```

## Critical Configuration Points:

- **Default Proceeding**: config.py:70 - DEFAULT_PROCEEDING = "R2207005"
- **Vector DB Path**: config.py:get_proceeding_file_paths()
- **Scraper Settings**: config.py:72-111 (Chrome options, timeouts)
- **Performance Settings**: config.py:116-128 (parallel workers, batch sizes)

## Progress Tracking:

- **0%**: "Starting CPUC RAG system..."
- **10%**: "Selecting default proceeding..."
- **20%**: "Initializing database and folders..."
- **30-65%**: "Running scraper workflow..." (varies by document count)
- **70-100%**: "Processing embeddings..." (incremental progress)

## Background Services Started:

1. **PDF Scheduler**: Checks for new documents every 12 hours
2. **Background Processor**: Processes new PDFs found during startup
3. **Document Scraper**: Maintains document discovery pipeline
4. **Real-time Notifications**: Updates UI with processing status

## Methods Called in Sequence:

1. **streamlit run app.py** → Framework initialization
2. **app.py:805 main()** → Application entry point
3. **app.py:813 execute_enhanced_startup_sequence()** → Cached startup
4. **startup_manager.py:380 create_startup_manager()** → Factory function
5. **startup_manager.py:46 execute_startup_sequence()** → Main coordination
6. **rag_core.py:34 CPUCRAGSystem.__init__()** → RAG system initialization
7. **cpuc_scraper.py:CPUCSimplifiedScraper.scrape_proceeding()** → **SIMPLIFIED** Document discovery
8. **data_processing.py:extract_and_chunk_with_docling_url()** → Document processing
9. **incremental_embedder.py:process_incremental_embeddings()** → Embedding creation
10. **app.py:875-936** → UI interface ready

## Simplified Scraper Workflow (UPDATED):

The scraper now follows an exact 7-step CPUC website navigation:

### **Step 1: Create Proceeding Folder**
- Creates `[proceeding]/` folder (e.g., `R2207005/`)

### **Step 2: CPUC Website Navigation** 
1. ➡️ Navigate to `https://apps.cpuc.ca.gov/apex/f?p=401:1`
2. 🔍 Enter proceeding in 'Proceeding Number Search:' text box  
3. 🔍 Click "Search" button
4. 📋 Click first result in "Proceeding Number" column
5. 📄 Click "Documents" tab
6. ⬇️ Click "Download" button (CSV export)
7. 💾 Rename downloaded `Documents.csv` to `[proceeding].csv`

### **Step 3: CSV Analysis**
- Extract PDFs from Document Type column
- Analyze with HTTP headers for metadata

### **Step 4: Google Search** 
- Top 10 results for `R.22-07-005 site:cpuc.ca.gov filetype:pdf`
- Only process URLs containing 'cpuc.ca.gov'

### **Step 5: Save History**
- `[proceeding]_scraped_pdf_history.json` with comprehensive metadata

**Real-time Progress Bar:**
```
🔍 Step 1: Navigating to CPUC main search page
🔍 Step 2: Entering proceeding number 'R2207005' in search box
🔍 Step 3: Clicking Search button
🔍 Step 4: Clicking on first proceeding result
🔍 Step 5: Clicking on Documents tab
🔍 Step 6: Clicking Download button
🔍 Step 7: Processing downloaded CSV file
Scraping R2207005 |████████████████████| 100.0% | Discovered: 45 | Scraped: 45
```

**File Structure Created:**
```
R2207005/
├── R2207005.csv                           # Official CPUC Documents.csv (renamed)
├── R2207005_scraped_pdf_history.json      # PDF metadata with analysis
```

**Technical Implementation:**
- **Selenium WebDriver**: Automated browser navigation
- **Chrome Download Preferences**: Automatic CSV download handling
- **XPath Selectors**: Robust element targeting (P5_PROCEEDING_SELECT, Documents tab)
- **Download Management**: File detection and renaming in download directory

**PDF Metadata Extracted:**
- pdf_url, title, document_type (from CPUC structure)
- pdf_creation_date (from HTTP Last-Modified headers)
- scrape_date, file_size, content_type
- Enhanced metadata: pages, author, subject

This comprehensive startup sequence ensures the CPUC RAG system is fully operational with document discovery, processing, embedding, and query capabilities before presenting the user interface.
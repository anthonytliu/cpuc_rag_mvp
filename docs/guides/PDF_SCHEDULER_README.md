# Automated PDF Update System

## Overview

The CPUC RAG system now includes an automated PDF update system that periodically checks for new documents and automatically updates the knowledge base. This system runs in the background and ensures that the latest regulatory documents are always available for analysis.

## Features

### 🔄 Automated PDF Checking
- Runs every 3 hours by default (configurable)
- Checks CPUC proceedings for new documents
- Downloads only new/updated PDFs to avoid redundant processing
- Maintains download history to track processed documents

### 🧠 Automatic RAG System Updates
- Automatically updates the vector store when new PDFs are downloaded
- Incremental processing - only processes new documents
- Maintains system availability during updates
- Updates timeline data with new document information

### 📊 Status Monitoring
- Real-time status display on the web interface
- Shows last check time and next scheduled check
- Displays download statistics and error logs
- Manual control buttons for immediate checks

### ⚙️ Configuration Options
- Configurable check intervals (default: 3 hours)
- Enable/disable scheduler via environment variables
- Headless or visible browser mode for scraping
- Configurable retry limits and timeouts

## Usage

### Web Interface

1. **Status Display**: The main page shows the current auto-update status and last check time
2. **System Status Tab**: View detailed scheduler information, statistics, and manual controls
3. **Manual Controls**: Force immediate PDF checks or refresh status

### Configuration

Set environment variables to customize behavior:

```bash
# Check interval (hours)
export PDF_CHECK_INTERVAL_HOURS=3

# Enable/disable scheduler
export PDF_SCHEDULER_ENABLED=true

# Headless mode for scraping
export PDF_SCHEDULER_HEADLESS=true

# Maximum retry attempts
export PDF_SCHEDULER_MAX_RETRIES=3

# Monitored proceedings (comma-separated)
export MONITORED_PROCEEDINGS=R2207005

# Auto-update RAG system
export AUTO_UPDATE_RAG_SYSTEM=true

# Delay before updating RAG (minutes)
export AUTO_UPDATE_DELAY_MINUTES=5
```

### Manual Testing

Test the scheduler functionality:

```bash
python test_scheduler.py
```

## Architecture

### Core Components

1. **PDFScheduler** (`pdf_scheduler.py`): Main scheduler class that manages background jobs
2. **CPUCPDFScraper** (`pdf_scraper_core.py`): Core PDF scraping functionality
3. **Background Jobs**: Scheduled tasks that run every N hours
4. **Status Tracking**: Persistent status file for monitoring

### Integration Points

- **RAG System**: Automatic vector store updates
- **Timeline System**: New documents appear in timeline
- **Streamlit App**: Status display and manual controls
- **Download History**: Prevents duplicate downloads

### Data Flow

1. **Scheduled Check**: Timer triggers PDF check
2. **Scraping**: Check CPUC website for new documents
3. **Download**: Download new PDFs to local storage
4. **RAG Update**: Automatically update vector store
5. **Status Update**: Save status and display in UI

## Files

### Main Files
- `pdf_scheduler.py`: Background job scheduler
- `pdf_scraper_core.py`: Core PDF scraping functionality
- `app.py`: Updated with scheduler integration
- `config.py`: Configuration settings

### Status Files
- `pdf_scheduler_status.json`: Scheduler status and statistics
- `cpuc_csvs/r2207005_scraped_pdf_history.json`: Scraped PDF history tracking

### Test Files
- `test_scheduler.py`: Comprehensive scheduler testing

## Monitoring

### Status Information
- Last check time
- Next scheduled check
- Download statistics
- Error logs
- RAG system update status

### Error Handling
- Automatic retries on failures
- Error logging with timestamps
- Graceful degradation (continues on partial failures)
- Status persistence across restarts

## Best Practices

### Production Deployment
1. Set `PDF_SCHEDULER_HEADLESS=true` for server environments
2. Monitor disk space for PDF storage
3. Set appropriate check intervals (3-6 hours recommended)
4. Monitor logs for errors and performance

### Development
1. Use `PDF_SCHEDULER_ENABLED=false` to disable during development
2. Set longer intervals for testing
3. Use visible browser mode for debugging scraping issues

### Maintenance
1. Monitor download history file size
2. Clean up old PDF files periodically
3. Monitor vector store size and performance
4. Review error logs regularly

## Troubleshooting

### Common Issues

1. **Scheduler Not Starting**
   - Check if Chrome/ChromeDriver is installed
   - Verify environment variables
   - Check system resources

2. **Download Failures**
   - Verify internet connection
   - Check CPUC website availability
   - Review retry settings

3. **RAG Update Issues**
   - Check vector store disk space
   - Monitor memory usage during updates
   - Verify file permissions

### Logs

Monitor application logs for:
- Scheduler start/stop events
- PDF check results
- Download progress
- RAG system updates
- Error messages

## Performance Considerations

- **Memory Usage**: RAG updates require significant memory
- **Disk Space**: Downloaded PDFs accumulate over time
- **Network**: Periodic scraping uses bandwidth
- **Processing Time**: Vector store updates take time

## Security

- **Headless Mode**: Recommended for production
- **Download Limits**: Respects server rate limits
- **Error Handling**: Prevents infinite retry loops
- **File Validation**: Validates downloaded PDFs

## Future Enhancements

- [ ] Email notifications for new documents
- [ ] Multiple proceeding support
- [ ] Advanced filtering options
- [ ] Performance optimizations
- [ ] Web API for external integrations
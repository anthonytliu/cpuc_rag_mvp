# Incremental Mode Fix - Data Deletion Prevention

## Problem
The incremental embedder was causing mass data deletion when processing individual documents. When `incremental_embedder.py` called `build_vector_store_from_urls()` with a single URL, the system interpreted all other previously processed URLs as "deleted" and removed their chunks from the vector store.

**Example of the problem:**
- Vector store contains 735 processed documents
- Incremental embedder processes 1 new document
- System sees only 1 URL in current batch vs 735 in storage
- System deletes 103,699 chunks for the "missing" 735 URLs
- Result: All existing data is lost!

## Root Cause
The `build_vector_store_from_urls()` method was designed for full synchronization scenarios where the provided URL list represents the complete desired state. It compared current URLs against stored URLs and deleted any that weren't in the current batch.

## Solution
Added an `incremental_mode` parameter to `build_vector_store_from_urls()`:

```python
def build_vector_store_from_urls(self, pdf_urls: List[Dict[str, str]], 
                                 force_rebuild: bool = False, 
                                 incremental_mode: bool = False):
```

### When `incremental_mode=True`:
- Only processes new URLs (adds them to the vector store)
- **Skips deletion logic entirely**
- Preserves all existing data
- Logs that deletions are being skipped

### When `incremental_mode=False` (default):
- Full synchronization mode
- Processes new URLs AND deletes removed URLs
- Maintains original behavior for bulk operations

## Changes Made

### 1. rag_core.py
- Added `incremental_mode` parameter to `build_vector_store_from_urls()`
- Modified deletion logic to respect incremental mode
- Enhanced logging to show which mode is active

### 2. incremental_embedder.py
- Updated to use `incremental_mode=True` when processing individual documents
- This prevents mass deletion during single-document processing

### 3. pdf_scheduler.py
- Updated to use `incremental_mode=True` when adding new URLs
- Ensures scheduled updates don't delete existing data

## Usage

### For individual document processing (incremental):
```python
# This is safe - won't delete existing data
rag_system.build_vector_store_from_urls([single_url], incremental_mode=True)
```

### For full synchronization:
```python
# This will sync the complete state - may delete removed URLs
rag_system.build_vector_store_from_urls(all_urls, incremental_mode=False)
```

## Testing
- Created test scripts to verify the fix works correctly
- Confirmed that incremental mode prevents deletions
- Confirmed that full sync mode still allows deletions when needed

## Backward Compatibility
The fix is fully backward compatible:
- Default behavior (`incremental_mode=False`) remains unchanged
- Existing code continues to work without modification
- Only code that needs incremental processing was updated to use the new parameter

This fix ensures that the data processor will **NEVER** delete existing files during incremental operations, only modify or append them as intended.
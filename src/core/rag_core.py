# 📁 rag_core.py
# The core RAG system logic.

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from langchain.chains import LLMChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.tools import DuckDuckGoSearchResults
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Try relative imports first, fall back to absolute
try:
    from . import config
    from ..data_processing import data_processing
    from . import models
    from . import utils
    from ..agents.response_agents import generate_multi_agent_response, AGENT_REGISTRY
    from ..agents.multi_agent_system import MultiAgentSystem, MultiAgentResult
except ImportError:
    # Fallback to absolute imports when running directly from src/
    from core import config
    from data_processing import data_processing
    from core import models
    from core import utils
    from agents.response_agents import generate_multi_agent_response, AGENT_REGISTRY
    from agents.multi_agent_system import MultiAgentSystem, MultiAgentResult

logger = logging.getLogger(__name__)


class CPUCRAGSystem:
    def __init__(self, current_proceeding: str = None):
        # --- Proceeding Configuration ---
        if current_proceeding is None:
            current_proceeding = config.DEFAULT_PROCEEDING
        
        self.current_proceeding = current_proceeding
        
        # Get proceeding-specific paths
        self.proceeding_paths = config.get_proceeding_file_paths(current_proceeding, config.PROJECT_ROOT)
        
        # --- Base Configuration ---
        self.num_chunks = None
        # Base directory no longer needed - using URL-based processing
        self.db_dir = self.proceeding_paths['vector_db']
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.project_root = config.PROJECT_ROOT

        # --- Models and Prompts ---
        self.embedding_model = models.get_embedding_model()
        self.llm = models.get_llm()
        self.technical_prompt = PromptTemplate.from_template(config.ACCURACY_PROMPT_TEMPLATE)
        # ENHANCEMENT: Initialize the new prompt and search tool
        self.layman_prompt = PromptTemplate.from_template(config.LAYMAN_PROMPT_TEMPLATE)
        self.search_tool = DuckDuckGoSearchResults()

        # --- State and Components ---
        self.vectordb: Optional[LanceDB] = None
        self.lance_db = None
        self.retriever = None
        self.doc_hashes_file = self.proceeding_paths['document_hashes']
        self.doc_hashes = self._load_doc_hashes()

        # --- Initial Setup ---
        self.db_dir.mkdir(exist_ok=True)
        
        # Load existing LanceDB vector store if it exists
        self._load_existing_lance_vector_store()
        
        logger.info(f"CPUCRAGSystem initialized. Processing mode: URL-based")
        if self.vectordb:
            try:
                # For LanceDB, check if it has a working connection
                if hasattr(self.vectordb, '_connection') and self.vectordb._connection is not None:
                    try:
                        # Try to get table info
                        table_names = self.vectordb._connection.table_names()
                        table_name = f"{self.current_proceeding}_documents"
                        
                        if table_name in table_names:
                            table = self.vectordb._connection.open_table(table_name)
                            chunk_count = len(table.to_pandas())
                            logger.info(f"Loaded existing LanceDB vector store with {chunk_count} chunks")
                            
                            # Set up QA pipeline since we have a working vector store with data
                            if chunk_count > 0:
                                self.setup_qa_pipeline()
                                logger.info("QA pipeline set up successfully with data")
                            else:
                                logger.warning("Vector store exists but is empty. QA pipeline not set up.")
                                logger.warning("Use standalone_data_processor.py to process documents and populate the vector store.")
                        else:
                            logger.warning("LanceDB table not found")
                            logger.warning("Use standalone_data_processor.py to create and populate the vector store.")
                            
                    except Exception as e:
                        logger.warning(f"Could not access LanceDB table: {e}")
                        logger.warning("Vector store may be corrupted or empty.")
                else:
                    logger.warning("LanceDB connection not properly initialized")
            except Exception as e:
                logger.warning(f"Vector store loaded but verification failed: {e}")
        else:
            logger.info("No working vector store found. Use standalone_data_processor.py to build vector store.")


    def query(self, question: str):
        """Enhanced query method with superseding logic."""
        if not self.retriever:
            yield "Error: RAG system's retriever is not ready."
            return

        yield "Step 1: Generating hypothetical answer for search..."
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that generates a hypothetical, detailed answer to the user's question. This answer will be used to find similar real documents."),
            ("human", "{question}")
        ])
        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        hypothetical_answer = hyde_chain.invoke({"question": question})

        base_retriever = self.vectordb.as_retriever(search_kwargs={"k": 30})  # Get more docs initially
        retrieved_docs = base_retriever.get_relevant_documents(hypothetical_answer)

        # Apply the enhanced reranking with recency and superseding logic
        reranked_docs = data_processing.rerank_documents_with_recency(self, question, retrieved_docs)

        final_docs = reranked_docs[:config.TOP_K_DOCS]
        logger.info(f"Using {len(final_docs)} final documents for context.")

        yield "Step 2: Synthesizing technical answer with citation placeholders..."
        default_no_answer = "The provided context does not contain sufficient information to answer this question."
        part1_answer = default_no_answer
        raw_part1_answer = ""

        if self.llm and final_docs:
            context = self._enhance_context_for_llm(final_docs, question)
            prompt = self.technical_prompt.format(context=context, question=question,
                                                  current_date=datetime.now().strftime("%B %d, %Y"))
            response_message = self.llm.invoke(prompt)
            raw_part1_answer = response_message.content

            # Post-process the raw answer to insert HTML links for citations
            part1_answer = self._add_inline_citations(raw_part1_answer, final_docs)

        yield "Step 3: Generating simplified explanation..."
        part2_summary = "A simplified explanation could not be generated."
        if self.llm and raw_part1_answer != default_no_answer:
            prompt = self.layman_prompt.format(technical_answer=raw_part1_answer)  # Use the raw answer for summary
            summary_message = self.llm.invoke(prompt)
            part2_summary = summary_message.content

        yield "Finalizing the response..."

        # Assemble the final answer with clean HTML.
        final_answer = self._assemble_final_answer(part1_answer, part2_summary)

        result_payload = {
            "answer": final_answer,
            "raw_part1_answer": raw_part1_answer,  # For debugging
            "sources": self._process_sources(final_docs),
            "confidence_indicators": self._assess_confidence(final_docs, raw_part1_answer, question)
        }
        yield result_payload

        ### FIX: New method to process citation placeholders and create HTML links.

    def _get_cpuc_url_from_filename(self, filename: str) -> str:
        """
        Maps a local PDF filename to its original CPUC URL or proceeding page.
        
        Since conference numbers in filenames don't directly map to DocIDs,
        this function provides useful CPUC links that help users find the documents.
        
        Args:
            filename (str): Local PDF filename
            
        Returns:
            str: CPUC URL or None if mapping fails
        """
        import re
        
        # Pattern 1: Files with Conference numbers - link to proceeding since we can't map directly
        conf_match = re.search(r'Conf#\s*(\d+)', filename)
        if conf_match:
            # Link to the main proceeding page where users can find all documents
            proceeding_urls = config.get_proceeding_urls(self.current_proceeding)
            return proceeding_urls['cpuc_apex']
        
        # Pattern 2: Numeric PDF files (like 498072273.PDF) - try as DocID
        numeric_match = re.match(r'^(\d+)\.PDF?$', filename, re.IGNORECASE)
        if numeric_match:
            doc_id = numeric_match.group(1)
            return f"https://docs.cpuc.ca.gov/SearchRes.aspx?DocFormat=ALL&DocID={doc_id}"
        
        # Pattern 3: For main proceeding document
        proceeding_formatted = config.format_proceeding_for_search(self.current_proceeding).lower()
        if any(keyword in filename.lower() for keyword in [proceeding_formatted, self.current_proceeding.lower(), 'oir']):
            proceeding_urls = config.get_proceeding_urls(self.current_proceeding)
            return proceeding_urls['cpuc_apex']
        
        # Pattern 4: Try to extract key terms for general CPUC search
        # Extract company/entity names for search
        for entity in ['PG&E', 'Pacific Gas', 'SCE', 'Southern California Edison', 'SDG&E', 'San Diego Gas']:
            if entity.lower() in filename.lower():
                search_term = entity.replace(' ', '+')
                proceeding_urls = config.get_proceeding_urls(self.current_proceeding)
                return f"{proceeding_urls['cpuc_search']}&searchfor={search_term}"
        
        # Fallback: Return None to use localhost
        return None

    def _add_inline_citations(self, text: str, source_documents: List[Document] = None) -> str:
        """
        Finds [CITE:...] placeholders and replaces them with HTML links to source URLs.
        
        This function processes the LLM output to convert citation placeholders
        into clickable links that open the source documents at the specified page.
        For URL-based processing, it links directly to the source URLs.
        For local PDFs, it maps filenames to CPUC URLs when possible.
        """
        def replace_match(match):
            filename = match.group("filename").strip()
            page = match.group("page").strip()
            
            # Priority 1: Try to find the source URL from document metadata (direct linkage)
            source_url = None
            if source_documents:
                for doc in source_documents:
                    doc_source = doc.metadata.get('source', '')
                    # Match by filename (with or without extensions)
                    filename_clean = filename.replace('.pdf', '').replace('.PDF', '')
                    doc_source_clean = doc_source.replace('.pdf', '').replace('.PDF', '')
                    
                    if (doc_source_clean == filename_clean or 
                        filename_clean in doc_source_clean or 
                        doc_source_clean in filename_clean):
                        source_url = doc.metadata.get('source_url')
                        if source_url:
                            logger.debug(f"Found source_url in metadata for {filename}: {source_url}")
                            break
            
            # Priority 2: If no source_url in metadata, try filename mapping (fallback)
            if not source_url:
                source_url = self._get_cpuc_url_from_filename(filename)
                if source_url:
                    logger.debug(f"Found source_url via filename mapping for {filename}: {source_url}")
            
            # Create final URL with page fragment
            if source_url:
                # For CPUC documents, add page fragment for direct navigation
                url = f"{source_url}#page={page}"
                logger.debug(f"Generated citation URL: {url}")
            else:
                # Priority 3: Fallback to localhost (for backward compatibility)
                url = f"http://localhost:{config.PDF_SERVER_PORT}/{filename}#page={page}"
                logger.debug(f"Using localhost fallback for {filename}")
            
            return f'<a href="{url}" target="_blank" title="Source: {filename}, Page: {page}" style="display: inline-block; text-decoration: none; font-size: 0.75em; font-weight: bold; color: #fff; background-color: #0d6efd; border-radius: 4px; padding: 2px 6px; margin-left: 3px; vertical-align: super;">[{page}]</a>'

        citation_pattern = re.compile(r"\[CITE:\s*(?P<filename>[^,]+),\s*page_(?P<page>\d+)\s*]")
        processed_text = citation_pattern.sub(replace_match, text)
        
        return processed_text if processed_text is not None else text

    def _assemble_final_answer(self, part1, part2) -> str:
        """
        Assembles the final HTML-formatted answer string with improved structure.
        
        This function combines the technical analysis and simplified explanation
        into a well-formatted, readable response with proper HTML styling.
        """
        return f"""<div style="background-color: #f8f9fa; border-radius: 8px; padding: 25px; margin-bottom: 25px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h3 style="border-bottom: 2px solid #0d6efd; padding-bottom: 10px; margin-top: 0; margin-bottom: 20px; color: #0d6efd; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        📋 Technical Analysis from Regulatory Documents
    </h3>
    <div style="line-height: 1.7; color: #333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        {part1}
    </div>
</div>

<div style="background-color: #f8f9fa; border-radius: 8px; padding: 25px; margin-bottom: 25px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h3 style="border-bottom: 2px solid #198754; padding-bottom: 10px; margin-top: 0; margin-bottom: 20px; color: #198754; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        💡 Simplified Explanation
    </h3>
    <div style="line-height: 1.7; color: #333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #e8f5e8; padding: 15px; border-radius: 6px; border-left: 4px solid #198754;">
        {part2}
    </div>
</div>"""

    def _rerank_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """Uses an LLM to re-rank documents based on their direct relevance to the query."""
        if not documents:
            return []

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful expert relevance-ranking assistant. Your task is to re-rank the following documents based on their relevance to the user's question. "
                "Output ONLY a comma-separated list of the document chunk_ids, from most relevant to least relevant. "
                "Pay close attention to specific numbers, dates, or names mentioned in the question."
            ),
            HumanMessagePromptTemplate.from_template(
                "QUESTION: {question}\n\n"
                "DOCUMENTS:\n{context}\n\n"
                "RANKED CHUNK_IDS:"
            )
        ])

        # Format the context with clear separators
        context_str = "\n---\n".join(
            [f"CHUNK_ID: {doc.metadata['chunk_id']}\nCONTENT: {doc.page_content}" for doc in documents])

        rerank_chain = prompt | self.llm | StrOutputParser()

        try:
            result = rerank_chain.invoke({"question": question, "context": context_str})
            ranked_ids = [id.strip() for id in result.split(',')]

            # Create a mapping of id to document for easy sorting
            doc_map = {doc.metadata['chunk_id']: doc for doc in documents}

            # Return documents in the new order, adding any that the LLM missed to the end
            sorted_docs = [doc_map[id] for id in ranked_ids if id in doc_map]

            # Add any documents that the LLM might have missed
            seen_ids = set(ranked_ids)
            sorted_docs.extend([doc for id, doc in doc_map.items() if id not in seen_ids])

            return sorted_docs
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}. Returning original order.")
            return documents

    def build_vector_store_from_urls(self, pdf_urls: List[Dict[str, str]], force_rebuild: bool = False, incremental_mode: bool = True):
        """
        Builds or incrementally updates the vector store using PDF URLs.
        
        This method processes PDF documents directly from URLs without downloading
        them locally. It maintains all incremental processing and caching logic
        while working entirely with URL-based processing.
        
        Args:
            pdf_urls (List[Dict[str, str]]): List of dictionaries containing:
                - 'url': PDF URL
                - 'title': Document title (optional)
                - 'id': Document ID for tracking (optional)
            force_rebuild (bool): Whether to force rebuild the entire vector store
            incremental_mode (bool): If True, only add new URLs without deleting existing ones
        """
        logger.info(f"Building vector store from {len(pdf_urls)} PDF URLs")
        
        # Handle force rebuild with safety guards
        if force_rebuild and self.db_dir.exists():
            logger.warning("🚨 FORCE REBUILD REQUESTED - This will delete all existing data!")
            logger.warning("🚨 This should only be used for data corruption or major schema changes")
            
            # Check if there's existing data
            try:
                if self.doc_hashes_file.exists():
                    existing_count = len(self.doc_hashes)
                    logger.warning(f"🚨 About to delete {existing_count} processed documents")
                    
                # Log the action for audit trail
                logger.warning(f"🚨 DELETING VECTOR STORE at {self.db_dir}")
                
                shutil.rmtree(self.db_dir)
                self.vectordb = None
                self.doc_hashes = {}
                if self.doc_hashes_file.exists():
                    self.doc_hashes_file.unlink()
                    
                logger.warning("🚨 Vector store deleted. Rebuilding from scratch.")
                    
            except Exception as e:
                logger.error(f"Failed to delete vector store during force rebuild: {e}")
                return
                
        # Initialize LanceDB vector store if needed
        if self.vectordb is None:
            try:
                # Initialize LanceDB connection if not already done
                if self.lance_db is None:
                    self.lance_db = lancedb.connect(str(self.db_dir))
                
                table_name = f"{self.current_proceeding}_documents"
                
                # Check if table exists
                existing_tables = self.lance_db.table_names()
                if table_name in existing_tables:
                    logger.info("Loading existing LanceDB table...")
                    self.vectordb = LanceDB(
                        connection=self.lance_db,
                        embedding=self.embedding_model,
                        table_name=table_name,
                        mode="append"  # Fix: Use append mode instead of default overwrite
                    )
                    logger.info("LanceDB vector store loaded successfully.")
                else:
                    logger.info("No existing LanceDB table found. Will create when first documents are added.")
                    
            except Exception as e:
                logger.error(f"Failed to initialize LanceDB: {e}")
                return

        # Process URLs - find new/updated ones
        current_url_hashes = {data_processing.get_url_hash(url_data['url']): url_data for url_data in pdf_urls}
        stored_url_hashes = set(self.doc_hashes.keys())
        
        new_url_hashes = set(current_url_hashes.keys()) - stored_url_hashes
        deleted_url_hashes = stored_url_hashes - set(current_url_hashes.keys())
        
        urls_to_process = [current_url_hashes[hash_val] for hash_val in new_url_hashes]
        
        # Log detailed statistics
        mode_str = "incremental" if incremental_mode else "full sync"
        logger.info(f"=== Vector Store URL Sync Statistics ({mode_str}) ===")
        logger.info(f"Total URLs provided: {len(pdf_urls)}")
        logger.info(f"Previously processed URLs: {len(stored_url_hashes)}")
        logger.info(f"New URLs to process: {len(urls_to_process)}")
        if not incremental_mode:
            logger.info(f"URLs to delete: {len(deleted_url_hashes)}")
        else:
            logger.info(f"URLs that would be deleted in full sync: {len(deleted_url_hashes)} (skipped in incremental mode)")
        
        # Debug: Show a few examples of hash comparison
        if len(current_url_hashes) > 0:
            sample_current = list(current_url_hashes.keys())[:3]
            sample_stored = list(stored_url_hashes)[:3]
            logger.debug(f"Sample current hashes: {sample_current}")
            logger.debug(f"Sample stored hashes: {sample_stored}")
        
        if urls_to_process:
            logger.info(f"URLs to process:")
            for url_data in urls_to_process:
                logger.info(f"  - {url_data.get('title', url_data['url'])}")
        
        # Process deletions (only in full sync mode, not incremental mode)
        if deleted_url_hashes and not incremental_mode:
            logger.info("Running full sync mode - processing deletions")
            self._delete_urls_from_db(deleted_url_hashes)
        elif deleted_url_hashes and incremental_mode:
            logger.info(f"Incremental mode enabled - skipping deletion of {len(deleted_url_hashes)} URLs")
            logger.debug("Use full sync mode to remove URLs that are no longer needed")

        # Process new/updated URLs
        if not urls_to_process:
            logger.info("✅ No new URLs to process. Vector store is up to date.")
            self.setup_qa_pipeline()
            return
        else:
            logger.info(f"🔄 Processing {len(urls_to_process)} new URLs with parallel processing.")
            start_time = datetime.now()
            all_new_chunks = []
            
            # Process URLs in parallel using ThreadPoolExecutor
            max_workers = min(config.URL_PARALLEL_WORKERS if hasattr(config, 'URL_PARALLEL_WORKERS') else 3, len(urls_to_process))
            logger.info(f"Using {max_workers} parallel workers for URL processing")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all URL processing tasks
                future_to_url = {
                    executor.submit(self._process_single_url, url_data): url_data 
                    for url_data in urls_to_process
                }
                
                # Process completed tasks with progress bar and incremental writes
                with tqdm(total=len(urls_to_process), desc="Processing URLs") as pbar:
                    for future in as_completed(future_to_url):
                        url_data = future_to_url[future]
                        pdf_url = url_data['url']
                        title = url_data.get('title', '')
                        
                        try:
                            result = future.result()
                            if result and result['chunks']:
                                # Use incremental write for immediate persistence
                                url_hash = data_processing.get_url_hash(pdf_url)
                                success = self.add_document_incrementally(
                                    chunks=result['chunks'],
                                    url_hash=url_hash,
                                    url_data=url_data,
                                    immediate_persist=True
                                )
                                
                                if success:
                                    all_new_chunks.extend(result['chunks'])
                                    logger.info(f"✅ Processed and persisted {len(result['chunks'])} chunks from {title or pdf_url}")
                                else:
                                    logger.error(f"❌ Failed to persist chunks from {title or pdf_url}")
                            else:
                                logger.warning(f"⚠️ No chunks extracted from {pdf_url}")
                                
                        except Exception as exc:
                            logger.error(f'URL {pdf_url} generated an exception during processing: {exc}', exc_info=True)
                        
                        pbar.update(1)

            # Final validation and cleanup
            if all_new_chunks:
                logger.info(f"Successfully processed {len(all_new_chunks)} new chunks with incremental writes")
                
                # Final persistence (redundant but ensures everything is saved)
                try:
                    if self.vectordb:
                        self.vectordb.persist()
                        logger.info("Final persistence completed successfully")
                except Exception as persist_exc:
                    logger.error(f"Final persistence failed: {persist_exc}")
                    
                # Final hash save (redundant but ensures everything is saved)
                try:
                    self._save_doc_hashes()
                    logger.info("Final hash save completed successfully")
                except Exception as hash_exc:
                    logger.error(f"Final hash save failed: {hash_exc}")
            else:
                logger.info("No new chunks were processed")

        # Note: Individual persistence already handled by incremental writes
        
        # Log final statistics with performance metrics
        end_time = datetime.now()
        processing_time = end_time - start_time
        total_chunks = len(all_new_chunks) if all_new_chunks else 0
        
        logger.info(f"=== URL Processing Performance Report ===")
        logger.info(f"Processing time: {processing_time}")
        logger.info(f"URLs processed: {len(urls_to_process)}")
        logger.info(f"Total chunks added: {total_chunks}")
        
        if len(urls_to_process) > 0:
            avg_time_per_url = processing_time.total_seconds() / len(urls_to_process)
            logger.info(f"Average time per URL: {avg_time_per_url:.2f} seconds")
            
        if total_chunks > 0:
            avg_time_per_chunk = processing_time.total_seconds() / total_chunks
            chunks_per_second = total_chunks / processing_time.total_seconds()
            logger.info(f"Average time per chunk: {avg_time_per_chunk:.3f} seconds")
            logger.info(f"Processing rate: {chunks_per_second:.2f} chunks/second")
            
        logger.info(f"Vector store batch size: {config.VECTOR_STORE_BATCH_SIZE}")
        logger.info(f"Parallel workers used: {max_workers}")
        logger.info(f"✅ URL-based database sync complete.")
        self.setup_qa_pipeline()

    def recover_partial_build(self, pdf_urls: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Recover from a partial build by identifying which URLs still need processing.
        
        This method compares the requested URLs with what's already in the vector store
        and returns only the URLs that still need to be processed.
        
        Args:
            pdf_urls: List of URL dictionaries to check
            
        Returns:
            List of URL dictionaries that still need processing
        """
        if not pdf_urls:
            return []
            
        logger.info("Checking for partial build recovery...")
        
        # Get current URL hashes
        current_url_hashes = {data_processing.get_url_hash(url_data['url']): url_data for url_data in pdf_urls}
        stored_url_hashes = set(self.doc_hashes.keys())
        
        # Find URLs that haven't been processed
        missing_url_hashes = set(current_url_hashes.keys()) - stored_url_hashes
        urls_to_process = [current_url_hashes[hash_val] for hash_val in missing_url_hashes]
        
        if urls_to_process:
            logger.info(f"Found {len(urls_to_process)} URLs that need processing for recovery")
            logger.info(f"Already processed: {len(stored_url_hashes)} URLs")
        else:
            logger.info("No URLs need processing - vector store is complete")
            
        return urls_to_process

    def create_checkpoint(self, checkpoint_name: str = None) -> str:
        """
        Create a checkpoint of the current vector store state.
        
        Args:
            checkpoint_name: Optional name for the checkpoint
            
        Returns:
            str: Path to the checkpoint directory
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        checkpoint_dir = self.db_dir.parent / f"checkpoints" / checkpoint_name
        
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy vector store files
            if self.db_dir.exists():
                shutil.copytree(self.db_dir, checkpoint_dir / "vector_store", dirs_exist_ok=True)
                
            # Copy document hashes
            if self.doc_hashes_file.exists():
                shutil.copy2(self.doc_hashes_file, checkpoint_dir / "document_hashes.json")
                
            logger.info(f"Created checkpoint: {checkpoint_dir}")
            return str(checkpoint_dir)
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None

    def restore_from_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Restore vector store from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            
        Returns:
            bool: True if restoration was successful
        """
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            # Stop current vector store
            self.vectordb = None
            
            # Restore vector store files
            vector_store_checkpoint = checkpoint_dir / "vector_store"
            if vector_store_checkpoint.exists():
                if self.db_dir.exists():
                    shutil.rmtree(self.db_dir)
                shutil.copytree(vector_store_checkpoint, self.db_dir)
                
            # Restore document hashes
            hash_checkpoint = checkpoint_dir / "document_hashes.json"
            if hash_checkpoint.exists():
                shutil.copy2(hash_checkpoint, self.doc_hashes_file)
                self.doc_hashes = self._load_doc_hashes()
                
            # Reload vector store
            self._load_existing_vector_store()
            
            logger.info(f"Successfully restored from checkpoint: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False

    def _normalize_document_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Normalize document metadata to ensure schema compatibility with existing LanceDB table.
        
        This function ensures all metadata fields are consistent and handle None values
        that could cause PyArrow schema casting issues.
        """
        normalized_docs = []
        
        for doc_idx, doc in enumerate(documents):
            try:
                normalized_metadata = {}
                
                # Safely copy all metadata and ensure None values become appropriate defaults
                for key, value in doc.metadata.items():
                    try:
                        if value is None:
                            normalized_metadata[key] = ""
                        elif isinstance(value, bool):
                            normalized_metadata[key] = value
                        elif isinstance(value, (int, float)):
                            # Handle NaN and infinite values
                            if str(value).lower() in ['nan', 'inf', '-inf']:
                                normalized_metadata[key] = 0
                            else:
                                normalized_metadata[key] = value
                        elif isinstance(value, str):
                            # Clean up string values
                            cleaned_value = value.strip()
                            normalized_metadata[key] = cleaned_value if cleaned_value else ""
                        else:
                            # Convert other types to string
                            normalized_metadata[key] = str(value)
                    except Exception as field_exc:
                        logger.warning(f"Failed to normalize field '{key}' in document {doc_idx}: {field_exc}")
                        normalized_metadata[key] = ""  # Use safe default
                
                # Ensure all expected fields exist with correct types and default values
                expected_fields = {
                    'source_url': '',
                    'source': '',
                    'page': 0,
                    'content_type': 'text',
                    'chunk_id': '',
                    'url_hash': '',
                    'last_checked': '',
                    'document_date': '',
                    'publication_date': '',
                    'proceeding_number': '',
                    'document_type': 'unknown',
                    'supersedes_priority': 0.5
                }
                
                # Remove fields that are known to cause schema incompatibility
                schema_incompatible_fields = [
                    'file_path', 'last_modified', 'extraction_method', 
                    'processing_status', 'chunking_strategy', 'file_size',
                    'mime_type', 'error_message', 'retry_count'
                ]
                for field in schema_incompatible_fields:
                    if field in normalized_metadata:
                        del normalized_metadata[field]
                
                # Add missing expected fields with defaults
                for field, default_value in expected_fields.items():
                    if field not in normalized_metadata:
                        normalized_metadata[field] = default_value
                    elif normalized_metadata[field] is None or normalized_metadata[field] == '':
                        if isinstance(default_value, (int, float)):
                            normalized_metadata[field] = default_value
                        else:
                            normalized_metadata[field] = default_value
                
                # Validate page content
                page_content = doc.page_content
                if page_content is None:
                    page_content = ""
                elif not isinstance(page_content, str):
                    page_content = str(page_content)
                
                # Ensure page content is not empty (LanceDB may reject empty content)
                if not page_content.strip():
                    page_content = "[Empty content]"
                
                normalized_docs.append(Document(
                    page_content=page_content,
                    metadata=normalized_metadata
                ))
                
            except Exception as doc_exc:
                logger.error(f"Failed to normalize document {doc_idx}: {doc_exc}")
                # Create a minimal fallback document to prevent complete failure
                fallback_doc = Document(
                    page_content="[Error processing document]",
                    metadata={
                        'source_url': '',
                        'source': f'error_doc_{doc_idx}',
                        'page': 0,
                        'content_type': 'text',
                        'chunk_id': f'error_{doc_idx}',
                        'url_hash': '',
                        'last_checked': '',
                        'document_date': '',
                        'publication_date': '',
                        'proceeding_number': '',
                        'document_type': 'error',
                        'supersedes_priority': 0.0
                    }
                )
                normalized_docs.append(fallback_doc)
        
        return normalized_docs

    def add_document_incrementally(self, chunks: List[Document], url_hash: str, url_data: Dict[str, str], 
                                  immediate_persist: bool = True) -> bool:
        """
        Add documents incrementally to the vector store with immediate persistence.
        
        This method writes documents to the DB immediately and handles failures gracefully,
        ensuring that progress is not lost during large batch operations.
        
        Args:
            chunks: List of Document objects to add
            url_hash: Hash of the URL being processed
            url_data: Dictionary containing URL metadata
            immediate_persist: Whether to persist immediately after adding
            
        Returns:
            bool: True if successful, False if failed
        """
        if not chunks:
            logger.warning("No chunks provided for incremental addition")
            return False
            
        # Ensure vector store is initialized
        if self.vectordb is None:
            logger.info("Initializing new LanceDB vector store for incremental addition.")
            try:
                # Initialize LanceDB connection if not already done
                if self.lance_db is None:
                    self.lance_db = lancedb.connect(str(self.db_dir))
                
                # Vector store will be created when first documents are added
                logger.info("LanceDB connection ready for document addition.")
            except Exception as e:
                logger.error(f"Failed to initialize LanceDB connection: {e}")
                return False
        
        try:
            # Add documents to vector store
            logger.info(f"Adding {len(chunks)} chunks incrementally...")
            
            # Process in smaller batches to avoid memory issues
            batch_size = min(config.VECTOR_STORE_BATCH_SIZE, len(chunks))
            success_count = 0
            
            # Create LanceDB vector store if it doesn't exist
            if self.vectordb is None and chunks:
                try:
                    # Let LangChain create the LanceDB table from documents
                    logger.info(f"Creating new LanceDB vector store with {len(chunks)} documents")
                    
                    # Normalize metadata before creating vector store
                    normalized_chunks = self._normalize_document_metadata(chunks)
                    
                    # Create vector store from documents
                    self.vectordb = LanceDB.from_documents(
                        documents=normalized_chunks,
                        embedding=self.embedding_model,
                        connection=self.lance_db,
                        table_name=f"{self.current_proceeding}_documents",
                        mode="append"  # Fix: Use append mode for future additions
                    )
                    
                    success_count = len(chunks)
                    logger.info(f"Created new LanceDB vector store with {len(chunks)} documents")
                    
                except Exception as create_exc:
                    logger.error(f"Failed to create LanceDB vector store: {create_exc}")
                    return False
            else:
                # Add to existing vector store in batches with improved error handling
                failed_batches = []
                total_batches = (len(chunks) - 1) // batch_size + 1
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    # Normalize metadata to ensure schema compatibility
                    try:
                        batch = self._normalize_document_metadata(batch)
                    except Exception as norm_exc:
                        logger.error(f"Failed to normalize batch {batch_num}: {norm_exc}")
                        failed_batches.append({'batch': batch_num, 'error': 'normalization_failed', 'exception': str(norm_exc)})
                        continue
                    
                    try:
                        self.vectordb.add_documents(documents=batch)
                        success_count += len(batch)
                        logger.debug(f"Successfully added batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                    except Exception as batch_exc:
                        error_msg = str(batch_exc)
                        
                        # Check for schema-related errors (critical failures)
                        if ("cast from string to null" in error_msg or 
                            "Field" in error_msg and "not found in target schema" in error_msg):
                            logger.warning(f"Schema compatibility issue detected: {batch_exc}")
                            logger.warning("The existing LanceDB table has an incompatible schema.")
                            logger.info("🔄 Attempting automatic schema migration...")
                            
                            # Attempt automatic schema migration
                            migration_success = self._attempt_schema_migration()
                            
                            if migration_success:
                                logger.info("✅ Schema migration successful, retrying batch addition...")
                                # Retry the failed batch after migration
                                try:
                                    self.vector_store.add_documents(batch, ids=batch_ids)
                                    successful_batches.append(batch_num)
                                    logger.info(f"✅ Successfully added batch {batch_num}/{total_batches} after migration")
                                    continue
                                except Exception as retry_exc:
                                    logger.error(f"❌ Batch addition failed even after migration: {retry_exc}")
                                    failed_batches.append({'batch': batch_num, 'error': 'migration_retry_failed', 'exception': str(retry_exc)})
                                    continue
                            else:
                                logger.error("❌ Schema migration failed. Manual intervention required.")
                                logger.error("To fix this manually, delete the existing table directory and rebuild:")
                                logger.error(f"rm -rf {self.db_dir}")
                                logger.error("Then run the data processor again to rebuild with correct schema.")
                                return False
                        else:
                            # Record non-critical failure and continue
                            logger.error(f"Failed to add batch {batch_num}/{total_batches}: {batch_exc}")
                            failed_batches.append({'batch': batch_num, 'error': 'add_failed', 'exception': str(batch_exc)})
                            continue
                
                # Report detailed results
                if failed_batches:
                    logger.warning(f"Failed to add {len(failed_batches)}/{total_batches} batches:")
                    for failure in failed_batches:
                        logger.warning(f"  Batch {failure['batch']}: {failure['error']} - {failure['exception']}")
                
                # Check if we had significant failures
                failure_rate = len(failed_batches) / total_batches
                if failure_rate > 0.5:  # More than 50% failed
                    logger.error(f"High failure rate: {len(failed_batches)}/{total_batches} batches failed ({failure_rate:.1%})")
                    logger.error("This indicates a systematic issue with the vector store or data processing")
            
            # For LanceDB, data is persisted automatically (no explicit persist needed)
            
            # Update document hashes and report results based on success/failure analysis
            total_chunks = len(chunks)
            success_rate = success_count / total_chunks if total_chunks > 0 else 0
            
            # Determine if operation was successful enough to continue
            operation_successful = success_count > 0 and success_rate >= 0.5  # At least 50% success rate
            
            if operation_successful:
                # Update document hashes with actual success count
                self.doc_hashes[url_hash] = {
                    'url': url_data['url'],
                    'title': url_data.get('title', ''),
                    'last_processed': datetime.now().isoformat(),
                    'chunk_count': success_count,
                    'total_chunks_processed': total_chunks,
                    'success_rate': f"{success_rate:.1%}"
                }
                
                # Save hashes immediately
                try:
                    self._save_doc_hashes()
                    logger.info(f"Document hashes updated for {url_data.get('title', url_data['url'])}")
                except Exception as hash_exc:
                    logger.error(f"Failed to save document hashes: {hash_exc}")
                    # Don't fail the entire operation for hash save failure
                
                if success_rate == 1.0:
                    logger.info(f"✅ Successfully added all {success_count}/{total_chunks} chunks")
                else:
                    logger.warning(f"⚠️ Partially successful: added {success_count}/{total_chunks} chunks ({success_rate:.1%} success rate)")
                
                return True
            else:
                if success_count == 0:
                    logger.error(f"❌ No chunks were successfully added (0/{total_chunks})")
                else:
                    logger.error(f"❌ Operation failed: only {success_count}/{total_chunks} chunks added ({success_rate:.1%} success rate)")
                    logger.error("This indicates systematic issues with vector store operations")
                
                return False
                
        except Exception as e:
            logger.error(f"Incremental document addition failed: {e}")
            return False

    def _process_single_url(self, url_data: Dict[str, str]) -> Dict:
        """
        Process a single URL and return the extracted chunks.
        
        This method is designed to be called in parallel for multiple URLs.
        It extracts and processes a PDF document from a URL and returns
        the results in a standardized format.
        
        Args:
            url_data (Dict[str, str]): Dictionary containing 'url' and optional 'title'
            
        Returns:
            Dict: Results containing 'chunks' list and processing metadata
        """
        pdf_url = url_data['url']
        title = url_data.get('title', '')
        
        try:
            doc_chunks = data_processing.extract_and_chunk_with_docling_url(pdf_url, title, self.current_proceeding)
            return {
                'chunks': doc_chunks,
                'url': pdf_url,
                'title': title,
                'success': True,
                'chunk_count': len(doc_chunks) if doc_chunks else 0
            }
        except Exception as e:
            logger.error(f"Failed to process URL {pdf_url}: {e}")
            return {
                'chunks': [],
                'url': pdf_url,
                'title': title,
                'success': False,
                'error': str(e),
                'chunk_count': 0
            }

    def _delete_urls_from_db(self, url_hashes_to_delete: set):
        """Finds and deletes all chunks associated with URL hashes."""
        if not url_hashes_to_delete:
            return

        logger.info(f"Found {len(url_hashes_to_delete)} URL hashes to delete from vector store.")
        
        # Get all records from the collection
        all_data = self.vectordb.get(include=["metadatas"])
        
        # Filter in Python to find the IDs to delete based on url_hash
        ids_to_delete = [
            record_id
            for record_id, metadata in zip(all_data['ids'], all_data['metadatas'])
            if metadata and metadata.get('url_hash') in url_hashes_to_delete
        ]

        if ids_to_delete:
            logger.info(f"Deleting {len(ids_to_delete)} chunks for {len(url_hashes_to_delete)} deleted URLs.")
            try:
                self.vectordb.delete(ids=ids_to_delete)
                # Update the hash map only after successful deletion from DB
                for url_hash in url_hashes_to_delete:
                    if url_hash in self.doc_hashes:
                        del self.doc_hashes[url_hash]
            except Exception as e:
                logger.error(f"Failed to delete chunks from the database: {e}")
        else:
            logger.warning("Found URLs marked for deletion, but no corresponding chunks were found in the DB.")

    def build_vector_store(self, force_rebuild: bool = False):
        """
        DEPRECATED: This method was for file-based processing. 
        Use build_vector_store_from_urls() for URL-based processing instead.
        
        This method now redirects to URL-based processing if a download history exists.
        """
        logger.warning("build_vector_store() is deprecated. Use build_vector_store_from_urls() instead.")
        
        # Check if we have a scraped PDF history to work with
        # Try both old and new naming conventions for backward compatibility
        download_history_path = self.proceeding_paths['scraped_pdf_history']
        if not download_history_path.exists():
            download_history_path = self.proceeding_paths['download_history']
        if download_history_path.exists():
            logger.info("Found download history, redirecting to URL-based processing...")
            try:
                with open(download_history_path, 'r') as f:
                    scraped_pdf_history = json.load(f)
                
                # Convert download history to URL format
                pdf_urls = []
                for hash_key, entry in download_history.items():
                    if isinstance(entry, dict) and entry.get('url') and entry.get('filename'):
                        pdf_urls.append({
                            'url': entry['url'],
                            'filename': entry['filename']
                        })
                
                if pdf_urls:
                    logger.info(f"Redirecting to URL-based processing with {len(pdf_urls)} URLs")
                    self.build_vector_store_from_urls(pdf_urls, force_rebuild)
                    return
                else:
                    logger.error("Download history exists but contains no valid URLs")
                    
            except Exception as e:
                logger.error(f"Failed to process download history: {e}")
        
        logger.error("No local PDFs or download history found. Please use build_vector_store_from_urls() with URL list.")
        return


    def setup_qa_pipeline(self):
        """
        Sets up a retrieval pipeline. Falls back to basic retriever if SelfQueryRetriever fails.
        """
        if self.vectordb is None:
            logger.error("Vector DB is not available. Cannot setup QA pipeline.")
            return

        try:
            # --- Define the metadata fields that the LLM can use to filter ---
            metadata_field_info = [
                AttributeInfo(
                    name="source",
                    description="The filename of the document, e.g., 'D.24-05-015 Decision.pdf'",
                    type="string",
                ),
                AttributeInfo(
                    name="page",
                    description="The page number within the document.",
                    type="integer",
                ),
                # You could add more, like 'content_type' if you wanted to query only tables.
            ]
            document_content_description = "Regulatory text from CPUC documents."

            # --- Try to set up the Self-Querying Retriever ---
            self.retriever = SelfQueryRetriever.from_llm(
                self.llm,
                self.vectordb,
                document_content_description,
                metadata_field_info,
                verbose=True,  # Set to True for great debugging logs
            )
            logger.info("Advanced Self-Querying QA pipeline is ready.")
            
        except Exception as e:
            logger.warning(f"SelfQueryRetriever setup failed: {e}")
            logger.info("Falling back to basic retriever...")
            
            # Fall back to basic retriever
            try:
                self.retriever = self.vectordb.as_retriever(search_kwargs={"k": config.TOP_K_DOCS})
                logger.info("Basic retriever QA pipeline is ready.")
            except Exception as fallback_e:
                logger.error(f"Even basic retriever setup failed: {fallback_e}")
                self.retriever = None


    def _validate_vector_store(self) -> bool:
        """
        Validate the loaded vector store for integrity and consistency.
        
        Performs basic health checks to ensure the vector store is functional
        and consistent with the hash file.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        if self.vectordb is None:
            return False
            
        try:
            # Check if we can access the collection
            collection_data = self.vectordb.get(limit=1)
            if collection_data is None:
                logger.warning("Vector store collection is None")
                return False
                
            # Check if collection count makes sense
            total_count = self.vectordb._collection.count()
            logger.info(f"Vector store contains {total_count} chunks")
            
            # Basic consistency check: if we have hashes but no vectors, something's wrong
            if len(self.doc_hashes) > 0 and total_count == 0:
                logger.warning("Hash file indicates processed documents but vector store is empty")
                return False
            
            # Additional check: Skip local PDF check since we moved to URL-based processing
            # if total_count == 0 and len(self.doc_hashes) == 0:
            #     logger.warning("Vector store and hash file are both empty - may need to rebuild from URLs")
            #     return False
                
            # Check if we can perform a simple query
            if total_count > 0:
                test_results = self.vectordb.similarity_search("test", k=1)
                if test_results is None:
                    logger.warning("Vector store failed basic similarity search")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Vector store validation failed: {e}")
            return False
    
    def _rebuild_corrupted_store(self):
        """
        Rebuild the vector store when corruption is detected.
        
        This method attempts to preserve as much data as possible before rebuilding.
        """
        logger.warning("🚨 CORRUPTION DETECTED - Attempting recovery...")
        
        # Preserve document hashes in memory before cleanup (no backup files)
        backup_created = False
        preserved_hashes = {}
        if self.doc_hashes_file.exists() and self.doc_hashes:
            try:
                preserved_hashes = self.doc_hashes.copy()
                logger.info("📦 Preserved document hashes in memory for recovery (no backup files created)")
                backup_created = True
            except Exception as e:
                logger.error(f"Failed to preserve document hashes: {e}")
        
        # Clean up corrupted database
        if self.db_dir.exists():
            logger.warning(f"🚨 Removing corrupted vector store at {self.db_dir}")
            shutil.rmtree(self.db_dir)
        
        # Reset state but preserve doc_hashes if preservation was successful
        self.vectordb = None
        if not backup_created:
            logger.warning("🚨 No hashes preserved - resetting document hashes")
            self.doc_hashes = {}
        else:
            self.doc_hashes = preserved_hashes
            logger.info("✅ Preserved document hashes for recovery")
            
        logger.warning("🚨 Corrupted vector store cleaned up. Recovery will use existing document hashes if available.")

    def _normalize_file_path(self, file_path: Path) -> str:
        """
        Normalize file path to be relative to project root for consistent hashing.
        
        This ensures that hash keys are consistent regardless of the absolute path
        of the project directory, making the system more portable.
        """
        try:
            # Convert to absolute path first, then make relative to project root
            abs_path = file_path.absolute()
            return str(abs_path.relative_to(self.project_root))
        except ValueError:
            # If file is outside project root, use absolute path
            return str(file_path.absolute())
    
    def _load_doc_hashes(self) -> Dict[str, str]:
        """
        Load document hashes from the JSON file.
        
        Automatically converts old absolute paths to relative paths for consistency.
        """
        if self.doc_hashes_file.exists():
            try:
                with open(self.doc_hashes_file, 'r') as f:
                    raw_hashes = json.load(f)
                    
                # Convert old absolute paths to relative paths
                normalized_hashes = {}
                for path_str, hash_val in raw_hashes.items():
                    path_obj = Path(path_str)
                    normalized_key = self._normalize_file_path(path_obj)
                    normalized_hashes[normalized_key] = hash_val
                
                return normalized_hashes
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load document hashes: {e}")
                return {}
        return {}

    def _save_doc_hashes(self):
        # Ensure the directory exists
        self.doc_hashes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.doc_hashes_file, 'w') as f: json.dump(self.doc_hashes, f, indent=2)

    def _load_existing_lance_vector_store(self):
        """
        Load existing LanceDB vector store during initialization.
        
        This method implements the proper LanceDB vector store loading logic:
        1. Check if lance_db folder exists and create if needed
        2. Check parity between scraped_pdf_history.json and document_hashes.json
        3. Load if parity exists, otherwise mark for rebuild
        """
        # Use the db_dir that was already correctly set in __init__ from config
        # No need to recalculate - it's already set to the correct standard location
        logger.info(f"Using standard LanceDB location: {self.db_dir}")
        
        try:
            self.db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"LanceDB directory ensured: {self.db_dir}")
        except Exception as e:
            logger.error(f"Failed to create LanceDB directory: {e}")
            return

        # Step 2: Initialize LanceDB connection
        try:
            self.lance_db = lancedb.connect(str(self.db_dir))
            logger.info(f"Connected to LanceDB at {self.db_dir}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            return

        # Check if we have an existing table
        table_name = f"{self.current_proceeding}_documents"
        try:
            existing_tables = self.lance_db.table_names()
            if table_name not in existing_tables:
                logger.info("No existing LanceDB table found. Will create new one when needed.")
                return
            
            # Create LangChain LanceDB wrapper for existing table
            self.vectordb = LanceDB(
                connection=self.lance_db,
                embedding=self.embedding_model,
                table_name=table_name,
                mode="append"  # Fix: Use append mode instead of default overwrite
            )
            
            logger.info(f"Loaded existing LanceDB table: {table_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load existing LanceDB table: {e}")
            self.vectordb = None
            
            # Validate the loaded store
            if self._validate_vector_store():
                logger.info("Successfully loaded and validated existing vector store.")
            else:
                logger.warning("Existing vector store failed validation. Will rebuild when needed.")
                self.vectordb = None
                
        except Exception as e:
            logger.error(f"Failed to load existing vector store: {e}")
            logger.info("Will create new vector store when needed.")
            self.vectordb = None

    def _check_vector_store_parity(self) -> Dict:
        """
        Check parity between scraped_pdf_history.json and document_hashes.json.
        
        Returns:
            Dict with keys:
                - has_parity: bool indicating if parity exists
                - reason: string explanation if parity fails
                - missing_files: list of files in scraped_pdf_history but not in document_hashes
                - extra_files: list of files in document_hashes but not in scraped_pdf_history
        """
        try:
            # Load scraped PDF history
            # Try both old and new naming conventions for backward compatibility
            download_history_path = self.proceeding_paths['scraped_pdf_history']
            if not download_history_path.exists():
                return {
                    'has_parity': False,
                    'reason': 'scraped_pdf_history.json not found',
                    'missing_files': [],
                    'extra_files': []
                }
            
            with open(download_history_path, 'r') as f:
                scraped_pdf_history = json.load(f)
            
            # Extract filenames from download history (only successfully downloaded files)
            scraped_pdf_history_files = set()
            for record in scraped_pdf_history.values():
                if record.get('status') == 'downloaded':
                    filename = record.get('filename', '')
                    if filename:
                        scraped_pdf_history_files.add(filename)
            
            # Extract filenames from document hashes (normalize paths)
            document_hashes_files = set()
            for path_str in self.doc_hashes.keys():
                # Extract filename from path
                path_obj = Path(path_str)
                filename = path_obj.name
                document_hashes_files.add(filename)
            
            # Check for mismatches
            missing_files = scraped_pdf_history_files - document_hashes_files
            extra_files = document_hashes_files - scraped_pdf_history_files
            
            has_parity = len(missing_files) == 0 and len(extra_files) == 0
            
            if has_parity:
                reason = "Perfect parity between scraped_pdf_history and document_hashes"
            else:
                reason = f"Parity mismatch: {len(missing_files)} missing, {len(extra_files)} extra"
            
            return {
                'has_parity': has_parity,
                'reason': reason,
                'missing_files': list(missing_files),
                'extra_files': list(extra_files)
            }
            
        except Exception as e:
            logger.error(f"Error checking vector store parity: {e}")
            return {
                'has_parity': False,
                'reason': f'Error checking parity: {str(e)}',
                'missing_files': [],
                'extra_files': []
            }


    def _process_sources(self, documents: List[Document]) -> List[Dict]:
        sources = []
        for i, doc in enumerate(documents):
            source = {
                "rank": i + 1, 
                "document": doc.metadata.get("source", "Unknown"),
                "proceeding": doc.metadata.get("proceeding", "Unknown"), 
                "page": doc.metadata.get("page", "Unknown"),
                "excerpt": doc.page_content[:400] + "...", 
                "relevance_score": doc.metadata.get("relevance_score", "N/A")
            }
            
            # Include enhanced citation metadata if available
            enhanced_fields = ['char_start', 'char_end', 'char_length', 'line_number', 
                             'line_range_end', 'text_snippet', 'token_count', 'chunk_level']
            
            for field in enhanced_fields:
                if field in doc.metadata:
                    source[field] = doc.metadata[field]
            
            sources.append(source)
            
        return sources

    def _attempt_schema_migration(self) -> bool:
        """
        Attempt to migrate the LanceDB schema to support enhanced citation metadata.
        
        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            logger.info("🔄 Starting schema migration for enhanced citation support...")
            
            # Step 1: No backup folder creation - direct migration with data preservation
            logger.info("📦 Starting direct schema migration (no backup folders created)")
            
            # Step 2: Remove the incompatible table
            table_path = self.db_dir / f"{self.current_proceeding}_documents.lance"
            if table_path.exists():
                logger.info(f"🗑️ Removing incompatible table: {table_path}")
                shutil.rmtree(table_path)
            
            # Step 3: Reinitialize vector store with new schema
            logger.info("🔄 Reinitializing vector store with enhanced schema...")
            self.vector_store = None
            self._load_existing_lance_vector_store()
            
            if self.vector_store:
                logger.info("✅ Schema migration completed successfully")
                return True
            else:
                logger.error("❌ Failed to reinitialize vector store after migration")
                return False
                
        except Exception as e:
            logger.error(f"💥 Schema migration failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _assess_confidence(self, documents: List[Document], answer: str, question: str) -> Dict:
        num_sources = len(documents)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        alignment_score = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
        score_factors = [
            (num_sources >= 3), (num_sources >= 5), utils.check_source_consistency(documents),
            (alignment_score > 0.3), (alignment_score > 0.5),
            bool(re.search(r'\[Source:.*?Page: \d+]', answer))
        ]
        confidence_score = sum(score_factors) / len(score_factors)
        if confidence_score > 0.8:
            confidence_level = "High"
        elif confidence_score > 0.5:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        return {
            "num_sources": num_sources, "source_consistency": utils.check_source_consistency(documents),
            "question_alignment": round(alignment_score, 2), "overall_confidence": confidence_level,
            "has_citations": '✅ Yes' if 'Source:' in answer else '❌ No',
        }


    def _enhance_context_for_llm(self, relevant_docs: List[Document], question: str) -> str:
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source_info = f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]"
            content = utils.extract_and_enhance_dates(doc.page_content)
            content = utils.highlight_regulatory_terms(content, question)
            context_parts.append(f"{source_info}\n{content}")
        return "\n\n---\n\n".join(context_parts)

    def get_system_stats(self) -> dict:
        """
        Get comprehensive system statistics including document counts and vector store status.
        
        Supports both file-based and URL-based processing modes based on configuration.
        
        Returns:
            dict: Statistics about the system state including processing mode and metrics
        """
        stats = {
            "embedding_model": config.EMBEDDING_MODEL_NAME, 
            "llm_model": config.OPENAI_MODEL_NAME,
            "processing_mode": "URL-based" if config.USE_URL_PROCESSING else "File-based",
            "total_urls_tracked": len(self.doc_hashes), 
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store_loaded": self.vectordb is not None
        }
        
        # URL-based processing stats (local PDFs removed)
        stats.update({
                "total_documents_on_disk": 0,  # No longer applicable
                "total_documents_hashed": len(self.doc_hashes),  # URLs processed
                "base_directory": "N/A (URL-based processing)",
                "base_directory_exists": False,
                "files_not_hashed": 0  # No longer applicable
            })
        
        # Vector store statistics (LanceDB)
        if self.vectordb:
            try:
                # For LanceDB, get chunk count from table
                if hasattr(self.vectordb, '_connection') and self.vectordb._connection is not None:
                    table_name = f"{self.current_proceeding}_documents"
                    table_names = self.vectordb._connection.table_names()
                    
                    if table_name in table_names:
                        table = self.vectordb._connection.open_table(table_name)
                        chunk_count = len(table.to_pandas())
                        stats["total_chunks"] = chunk_count
                        stats["vector_store_status"] = "loaded"
                    else:
                        stats["total_chunks"] = 0
                        stats["vector_store_status"] = "table_not_found"
                else:
                    stats["total_chunks"] = 0
                    stats["vector_store_status"] = "connection_error"
                
                # Calculate processing efficiency
                if len(self.doc_hashes) > 0 and stats["total_chunks"] > 0:
                    stats["avg_chunks_per_document"] = stats["total_chunks"] / len(self.doc_hashes)
                else:
                    stats["avg_chunks_per_document"] = 0
                    
            except Exception as e:
                stats["total_chunks"] = 0
                stats["vector_store_status"] = f"error: {str(e)}"
                stats["avg_chunks_per_document"] = 0
        else:
            stats["total_chunks"] = 0
            stats["vector_store_status"] = "not_loaded"
            stats["avg_chunks_per_document"] = 0
        
        return stats

# 📁 rag_core.py
# The core RAG system logic.

import json
import logging
import multiprocessing
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
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
import data_processing
import models
import utils
from memory import ConversationMemory

logger = logging.getLogger(__name__)


class CPUCRAGSystem:
    def __init__(self):
        # --- Base Configuration ---
        self.num_chunks = None
        self.base_dir = config.BASE_PDF_DIR
        self.db_dir = config.DB_DIR
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
        self.conversation_memory = ConversationMemory()
        self.vectordb: Optional[Chroma] = None
        self.retriever = None
        self.doc_hashes_file = self.db_dir / "document_hashes.json"
        self.doc_hashes = self._load_doc_hashes()

        # --- Initial Setup ---
        self.db_dir.mkdir(exist_ok=True)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base PDF directory does not exist: {self.base_dir}")
        
        # Load existing vector store if it exists
        self._load_existing_vector_store()
        
        logger.info(f"CPUCRAGSystem initialized. PDF source: {self.base_dir.absolute()}")
        if self.vectordb:
            try:
                chunk_count = self.vectordb._collection.count()
                logger.info(f"Loaded existing vector store with {chunk_count} chunks")
                # Set up QA pipeline since we have a working vector store
                if chunk_count > 0:
                    self.setup_qa_pipeline()
                else:
                    logger.info("Vector store exists but is empty. QA pipeline not set up.")
            except Exception as e:
                logger.warning(f"Vector store loaded but chunk count failed: {e}")
        else:
            logger.info("No existing vector store found. Use build_vector_store() to create one.")

    def has_new_pdfs(self) -> bool:
        """
        Check if there are new PDFs that haven't been processed yet.
        Returns True if new PDFs are found, False otherwise.
        
        This method is designed to be called by external systems (like PDF scrapers)
        to determine if a vector store rebuild is needed.
        """
        try:
            current_pdf_paths = {p for p in self.base_dir.rglob("*.pdf")}
            if not current_pdf_paths:
                logger.info("No PDFs found on disk")
                return False
                
            # Check for files that need processing
            files_to_process = [p for p in current_pdf_paths if self._needs_update(p)]
            
            if files_to_process:
                logger.info(f"Found {len(files_to_process)} new/updated PDFs that need processing")
                return True
            else:
                logger.info("All PDFs are up to date in vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error checking for new PDFs: {e}")
            return False

    def sync_if_needed(self) -> bool:
        """
        Automatically sync vector store if new PDFs are detected.
        Returns True if sync was performed, False if no sync was needed.
        
        This is the preferred method for automated systems to call.
        """
        if self.has_new_pdfs():
            logger.info("New PDFs detected. Starting incremental sync...")
            self.build_vector_store()
            return True
        else:
            logger.info("No new PDFs detected. Sync not needed.")
            return False

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

    def _add_inline_citations(self, text: str, source_documents: List[Document] = None) -> str:
        """
        Finds [CITE:...] placeholders and replaces them with HTML links to source URLs.
        
        This function processes the LLM output to convert citation placeholders
        into clickable links that open the source documents at the specified page.
        For URL-based processing, it links directly to the source URLs.
        """
        def replace_match(match):
            filename = match.group("filename").strip()
            page = match.group("page").strip()
            
            # Try to find the source URL from document metadata
            source_url = None
            if source_documents:
                for doc in source_documents:
                    if doc.metadata.get('source', '').replace('.pdf', '') in filename or filename in doc.metadata.get('source', ''):
                        source_url = doc.metadata.get('source_url')
                        if source_url:
                            break
            
            # Use source URL if available, otherwise fall back to localhost
            if source_url:
                # For CPUC documents, construct direct PDF URL with page fragment
                url = f"{source_url}#page={page}"
            else:
                # Fallback to localhost (for backward compatibility)
                url = f"http://localhost:{config.PDF_SERVER_PORT}/{filename}#page={page}"
            
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

    def build_vector_store_from_urls(self, pdf_urls: List[Dict[str, str]], force_rebuild: bool = False):
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
        """
        logger.info(f"Building vector store from {len(pdf_urls)} PDF URLs")
        
        # Handle force rebuild
        if force_rebuild and self.db_dir.exists():
            logger.warning("Force rebuild requested. Deleting existing vector store.")
            shutil.rmtree(self.db_dir)
            self.vectordb = None
            self.doc_hashes = {}
            if self.doc_hashes_file.exists():
                self.doc_hashes_file.unlink()
                
        # Initialize vector store if needed
        if self.vectordb is None:
            if self.db_dir.exists() and any(self.db_dir.iterdir()):
                logger.info("Loading existing vector store...")
                try:
                    self.vectordb = Chroma(persist_directory=str(self.db_dir), embedding_function=self.embedding_model)
                    
                    if self._validate_vector_store():
                        logger.info("Vector store loaded successfully and passed health checks.")
                    else:
                        logger.warning("Vector store loaded but failed health checks. Rebuilding...")
                        self._rebuild_corrupted_store()
                        
                except Exception as e:
                    logger.error(f"Failed to load existing DB, it might be corrupt. Rebuilding. Error: {e}")
                    self._rebuild_corrupted_store()
                    
            if self.vectordb is None:
                logger.info("Initializing new vector store.")
                self.vectordb = Chroma(embedding_function=self.embedding_model, persist_directory=str(self.db_dir))

        # Process URLs - find new/updated ones
        current_url_hashes = {data_processing.get_url_hash(url_data['url']): url_data for url_data in pdf_urls}
        stored_url_hashes = set(self.doc_hashes.keys())
        
        new_url_hashes = set(current_url_hashes.keys()) - stored_url_hashes
        deleted_url_hashes = stored_url_hashes - set(current_url_hashes.keys())
        
        urls_to_process = [current_url_hashes[hash_val] for hash_val in new_url_hashes]
        
        # Log detailed statistics
        logger.info(f"=== Vector Store URL Sync Statistics ===")
        logger.info(f"Total URLs provided: {len(pdf_urls)}")
        logger.info(f"Previously processed URLs: {len(stored_url_hashes)}")
        logger.info(f"New URLs to process: {len(urls_to_process)}")
        logger.info(f"URLs to delete: {len(deleted_url_hashes)}")
        
        if urls_to_process:
            logger.info(f"URLs to process:")
            for url_data in urls_to_process:
                logger.info(f"  - {url_data.get('title', url_data['url'])}")
        
        # Process deletions
        if deleted_url_hashes:
            self._delete_urls_from_db(deleted_url_hashes)

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
                
                # Process completed tasks with progress bar
                with tqdm(total=len(urls_to_process), desc="Processing URLs") as pbar:
                    for future in as_completed(future_to_url):
                        url_data = future_to_url[future]
                        pdf_url = url_data['url']
                        title = url_data.get('title', '')
                        
                        try:
                            result = future.result()
                            if result and result['chunks']:
                                all_new_chunks.extend(result['chunks'])
                                url_hash = data_processing.get_url_hash(pdf_url)
                                self.doc_hashes[url_hash] = {
                                    'url': pdf_url,
                                    'title': title,
                                    'last_processed': datetime.now().isoformat(),
                                    'chunk_count': len(result['chunks'])
                                }
                                logger.info(f"✅ Processed {len(result['chunks'])} chunks from {title or pdf_url}")
                            else:
                                logger.warning(f"⚠️ No chunks extracted from {pdf_url}")
                                
                        except Exception as exc:
                            logger.error(f'URL {pdf_url} generated an exception during processing: {exc}', exc_info=True)
                        
                        pbar.update(1)

            # Add new chunks to the database in batches
            if all_new_chunks:
                logger.info(f"Adding {len(all_new_chunks)} new chunks to the vector store...")
                
                db_batch_size = config.VECTOR_STORE_BATCH_SIZE
                logger.info(f"Using optimized batch size: {db_batch_size}")
                for i in tqdm(range(0, len(all_new_chunks), db_batch_size), desc="Writing to VectorDB"):
                    batch = all_new_chunks[i:i + db_batch_size]
                    try:
                        self.vectordb.add_documents(documents=batch)
                    except Exception as db_exc:
                        logger.error(f"Failed to add a batch of {len(batch)} chunks to DB: {db_exc}")

                logger.info("Finished adding new chunks.")

        # Persist all changes and save hashes
        logger.info("Persisting all database changes and saving URL hashes...")
        self.vectordb.persist()
        self._save_doc_hashes()
        
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
            doc_chunks = data_processing.extract_and_chunk_with_docling_url(pdf_url, title)
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
        Builds or incrementally updates the vector store in parallel.
        Processes and saves progress one file at a time.
        """
        # ... (force_rebuild and DB loading logic is unchanged)
        if force_rebuild and self.db_dir.exists():
            logger.warning("Force rebuild requested. Deleting existing vector store.")
            shutil.rmtree(self.db_dir)
            self.vectordb = None
            self.doc_hashes = {}
            if self.doc_hashes_file.exists():
                self.doc_hashes_file.unlink()
        if self.vectordb is None:
            if self.db_dir.exists() and any(self.db_dir.iterdir()):
                logger.info("Loading existing vector store...")
                try:
                    self.vectordb = Chroma(persist_directory=str(self.db_dir), embedding_function=self.embedding_model)
                    
                    # Perform health check on loaded database
                    if self._validate_vector_store():
                        logger.info("Vector store loaded successfully and passed health checks.")
                    else:
                        logger.warning("Vector store loaded but failed health checks. Rebuilding...")
                        self._rebuild_corrupted_store()
                        
                except Exception as e:
                    logger.error(f"Failed to load existing DB, it might be corrupt. Rebuilding. Error: {e}")
                    self._rebuild_corrupted_store()
                    
            if self.vectordb is None:
                logger.info("Initializing new vector store.")
                self.vectordb = Chroma(embedding_function=self.embedding_model, persist_directory=str(self.db_dir))

        # --- Sync Files ---
        current_pdf_paths = {p for p in self.base_dir.rglob("*.pdf")}
        
        # Convert stored paths (normalized) back to actual paths for comparison
        stored_pdf_paths = set()
        for normalized_path in self.doc_hashes.keys():
            try:
                # Try to resolve relative path to actual path
                actual_path = self.project_root / normalized_path
                if actual_path.exists():
                    stored_pdf_paths.add(actual_path)
            except Exception:
                # If normalization fails, skip this entry
                continue
        
        files_to_process = [p for p in current_pdf_paths if self._needs_update(p)]
        files_to_delete = stored_pdf_paths - current_pdf_paths

        # Log detailed statistics
        logger.info(f"=== Vector Store Sync Statistics ===")
        logger.info(f"Current PDFs found: {len(current_pdf_paths)}")
        logger.info(f"Previously processed: {len(stored_pdf_paths)}")
        logger.info(f"Files to process: {len(files_to_process)}")
        logger.info(f"Files to delete: {len(files_to_delete)}")
        
        if files_to_process:
            logger.info(f"Files to process:")
            for file_path in files_to_process:
                logger.info(f"  - {file_path.name}")
        
        if files_to_delete:
            logger.info(f"Files to delete:")
            for file_path in files_to_delete:
                logger.info(f"  - {file_path.name}")

        # --- Process Deletions ---
        if files_to_delete:
            self._delete_files_from_db(files_to_delete)

        # --- Process New/Modified Files ---
        if not files_to_process:
            logger.info("✅ No new or modified files to process. Vector store is up to date.")
            self.setup_qa_pipeline()
            return
        else:
            logger.info(f"🔄 Processing {len(files_to_process)} new/modified files to process.")
            start_time = datetime.now()
            all_new_chunks = []
            max_workers = min(10, multiprocessing.cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(data_processing.extract_and_chunk_with_docling, pdf_path): pdf_path for
                           pdf_path in files_to_process}

                progress_bar = tqdm(as_completed(futures), total=len(files_to_process), desc="Processing PDFs")
                processed_count = 0
                save_interval = 10  # Save every 10 PDFs processed
                
                for future in progress_bar:
                    pdf_path = futures[future]
                    progress_bar.set_description(f"Processing {pdf_path.name}")
                    try:
                        doc_chunks = future.result()
                        if doc_chunks:
                            # Immediately save chunks to vector store
                            db_batch_size = 64
                            chunks_saved = 0
                            for i in range(0, len(doc_chunks), db_batch_size):
                                batch = doc_chunks[i:i + db_batch_size]
                                try:
                                    self.vectordb.add_documents(documents=batch)
                                    chunks_saved += len(batch)
                                except Exception as db_exc:
                                    logger.error(f"Failed to save batch for {pdf_path.name}: {db_exc}")
                            
                            if chunks_saved > 0:
                                # Only save hash if chunks were successfully saved
                                normalized_path = self._normalize_file_path(pdf_path)
                                self.doc_hashes[normalized_path] = data_processing.get_file_hash(pdf_path)
                                all_new_chunks.extend(doc_chunks)
                                processed_count += 1
                                
                                # Progressive hash saving every 10 files
                                if processed_count % save_interval == 0:
                                    try:
                                        self._save_doc_hashes()
                                        logger.info(f"Progressive save: {processed_count} PDFs processed, hashes saved")
                                    except Exception as save_exc:
                                        logger.error(f"Failed to save hashes progressively: {save_exc}")
                    except Exception as exc:
                        logger.error(f'{pdf_path.name} generated an exception during processing: {exc}', exc_info=True)

            logger.info(f"Progressive processing complete. Total chunks processed: {len(all_new_chunks)}")

        # --- Persist all changes and save hashes ---
        logger.info("Persisting all database changes and saving file hashes...")
        self.vectordb.persist()
        self._save_doc_hashes()  # Save all hashes at the end of a successful sync.
        
        # Log final statistics
        end_time = datetime.now()
        processing_time = end_time - start_time
        logger.info(f"=== Processing Complete ===")
        logger.info(f"Processing time: {processing_time}")
        logger.info(f"Files processed: {len(files_to_process)}")
        logger.info(f"Total chunks added: {len(all_new_chunks) if all_new_chunks else 0}")
        logger.info(f"✅ Database sync complete.")
        self.setup_qa_pipeline()

        ### FIX: A more robust way to handle deletions that avoids complex `where` filters.

    def _delete_files_from_db(self, files_to_delete: set[Path]):
        """Finds and deletes all chunks associated with a set of file paths."""
        if not files_to_delete:
            return

        logger.info(f"Found {len(files_to_delete)} files to delete from vector store.")
        source_names_to_delete = {p.name for p in files_to_delete}

        # 1. Get ALL records from the collection.
        # For very large databases, this can be memory intensive. An alternative is to
        # query for each file to delete, but this is much faster for a moderate number of deletions.
        all_data = self.vectordb.get(include=["metadatas"])

        # 2. Filter in Python to find the IDs to delete.
        ids_to_delete = [
            record_id
            for record_id, metadata in zip(all_data['ids'], all_data['metadatas'])
            if metadata and metadata.get('source') in source_names_to_delete
        ]

        if ids_to_delete:
            logger.info(f"Deleting {len(ids_to_delete)} chunks for {len(files_to_delete)} deleted files.")
            try:
                self.vectordb.delete(ids=ids_to_delete)
                # Update the hash map only after successful deletion from DB.
                for pdf_path in files_to_delete:
                    normalized_path = self._normalize_file_path(pdf_path)
                    if normalized_path in self.doc_hashes:
                        del self.doc_hashes[normalized_path]
            except Exception as e:
                logger.error(f"Failed to delete chunks from the database: {e}")
        else:
            logger.warning("Found files marked for deletion, but no corresponding chunks were found in the DB.")

    # The helper _needs_update is still required.
    def _needs_update(self, file_path: Path) -> bool:
        """
        Checks if a file is new or has been modified since the last run.
        
        Uses normalized paths for consistent tracking across different environments.
        """
        current_hash = data_processing.get_file_hash(file_path)
        if not current_hash:
            logger.warning(f"Failed to calculate hash for {file_path}")
            return False
        
        normalized_path = self._normalize_file_path(file_path)
        stored_hash = self.doc_hashes.get(normalized_path)
        
        if stored_hash is None:
            logger.debug(f"New file detected: {normalized_path}")
            return True
        
        if current_hash != stored_hash:
            logger.debug(f"Modified file detected: {normalized_path}")
            return True
        
        logger.debug(f"File unchanged: {normalized_path}")
        return False

    def setup_qa_pipeline(self):
        """
        Sets up an advanced retrieval pipeline using a Self-Querying Retriever.
        This allows the LLM to write its own filters based on the user's query.
        """
        if self.vectordb is None:
            logger.error("Vector DB is not available. Cannot setup QA pipeline.")
            return

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

        # --- Set up the Self-Querying Retriever ---
        # It uses the LLM to translate a user's question into a structured query
        self.retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.vectordb,
            document_content_description,
            metadata_field_info,
            verbose=True,  # Set to True for great debugging logs
        )
        logger.info("Advanced Self-Querying QA pipeline is ready.")


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
            
            # Additional check: if we have many PDFs on disk but no chunks AND no hashes, validation should fail
            # Only fail if BOTH conditions are true: no chunks AND no hash tracking
            pdf_count = len(list(self.pdf_dir.glob("*.pdf")))
            if pdf_count > 0 and total_count == 0 and len(self.doc_hashes) == 0:
                logger.warning(f"Found {pdf_count} PDFs on disk but vector store and hash file are both empty")
                return False
                
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
        
        Cleans up corrupted files and resets the system state.
        """
        logger.info("Rebuilding corrupted vector store...")
        
        # Clean up corrupted database
        if self.db_dir.exists():
            shutil.rmtree(self.db_dir)
        
        # Reset state
        self.vectordb = None
        self.doc_hashes = {}
        
        # Remove corrupted hash file
        if self.doc_hashes_file.exists():
            self.doc_hashes_file.unlink()
            
        logger.info("Corrupted vector store cleaned up. Will rebuild from scratch.")

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

    def _load_existing_vector_store(self):
        """
        Load existing vector store during initialization if it exists and is valid.
        
        This method checks for an existing ChromaDB and loads it if found,
        preventing the need to reprocess all documents on every startup.
        """
        if not self.db_dir.exists() or not any(self.db_dir.iterdir()):
            logger.info("No existing vector store found. Will create new one when needed.")
            return
            
        logger.info("Found existing vector store directory. Attempting to load...")
        
        try:
            # Try to load the existing vector store
            self.vectordb = Chroma(
                persist_directory=str(self.db_dir), 
                embedding_function=self.embedding_model
            )
            
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

    def _process_sources(self, documents: List[Document]) -> List[Dict]:
        return [{
            "rank": i + 1, "document": doc.metadata.get("source", "Unknown"),
            "proceeding": doc.metadata.get("proceeding", "Unknown"), "page": doc.metadata.get("page", "Unknown"),
            "excerpt": doc.page_content[:400] + "...", "relevance_score": doc.metadata.get("relevance_score", "N/A")
        } for i, doc in enumerate(documents)]

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
        
        # Add file-based stats for backward compatibility
        if hasattr(self, 'base_dir') and self.base_dir and self.base_dir.exists():
            current_pdf_files = list(self.base_dir.rglob("*.pdf"))
            stats.update({
                "total_documents_on_disk": len(current_pdf_files),
                "base_directory": str(self.base_dir),
                "base_directory_exists": True,
                "files_not_processed": len(current_pdf_files) - len(self.doc_hashes)
            })
        else:
            stats.update({
                "total_documents_on_disk": 0,
                "base_directory": "N/A (URL-based processing)",
                "base_directory_exists": False,
                "files_not_processed": 0
            })
        
        # Vector store statistics
        if self.vectordb:
            try:
                chunk_count = self.vectordb._collection.count()
                stats["total_chunks"] = chunk_count
                stats["vector_store_status"] = "loaded"
                
                # Calculate processing efficiency
                if len(self.doc_hashes) > 0:
                    stats["avg_chunks_per_document"] = chunk_count / len(self.doc_hashes)
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

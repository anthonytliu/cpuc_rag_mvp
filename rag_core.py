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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        logger.info(f"CPUCRAGSystem initialized. PDF source: {self.base_dir.absolute()}")

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
            part1_answer = self._add_inline_citations(raw_part1_answer)

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

    def _add_inline_citations(self, text: str) -> str:
        """Finds [CITATION:...] placeholders and replaces them with HTML links."""

        # Define the replacement function at the correct scope (top-level inside the method)
        def replace_match(match):
            filename = match.group("filename").strip()
            # The page number in the prompt is now just a number, not "page_N"
            page = match.group("page").strip()

            # The URL points to our local PDF server.
            # Adobe's format for opening a PDF to a specific page is #page=N
            url = f"http://localhost:{config.PDF_SERVER_PORT}/{filename}#page={page}"

            # The HTML for the clickable, styled citation
            # The title attribute provides the hover-over text.
            return f"""
    <a href="{url}" target="_blank" title="Source: {filename}, Page: {page}" style="
        text-decoration: none;
        font-size: 0.75em;
        font-weight: bold;
        color: #fff;
        background-color: #0d6efd;
        border-radius: 4px;
        padding: 2px 6px;
        margin-left: 3px;
        vertical-align: super;
    ">[{page}]</a>
    """

        # Regex to find all placeholders, capturing filename and page
        # Updated to match the prompt's `page_12` format.
        citation_pattern = re.compile(r"\[CITATION:\s*(?P<filename>[^,]+),\s*page_(?P<page>\d+)\s*]")

        # Use re.sub with the now-defined replacement function to process all matches
        processed_text = citation_pattern.sub(replace_match, text)

        # Ensure the function always returns a string and handle newlines for HTML
        final_text = processed_text if processed_text is not None else text
        return final_text.replace("\n", "<br>")

    def _assemble_final_answer(self, part1, part2) -> str:
        """Assembles the final HTML-formatted answer string."""
        return f"""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #dee2e6;">
            <h3 style="border-bottom: 2px solid #0d6efd; padding-bottom: 10px; margin-top: 0; color: #0d6efd;">Part 1: Direct Answer from Corpus</h3>
            <p style="white-space: pre-wrap; line-height: 1.6;">{part1}</p>
        </div>
    
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #dee2e6;">
            <h3 style="border-bottom: 2px solid #198754; padding-bottom: 10px; margin-top: 0; color: #198754;">Part 2: Simplified Explanation</h3>
            <p style="white-space: pre-wrap; line-height: 1.6;">{part2}</p>
        </div>
        """

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
                except Exception as e:
                    logger.error(f"Failed to load existing DB, it might be corrupt. Rebuilding. Error: {e}")
                    shutil.rmtree(self.db_dir)
                    self.vectordb = None
                    self.doc_hashes = {}
            if self.vectordb is None:
                logger.info("Initializing new vector store.")
                self.vectordb = Chroma(embedding_function=self.embedding_model, persist_directory=str(self.db_dir))

        # --- Sync Files ---
        current_pdf_paths = {p for p in self.base_dir.rglob("*.pdf")}
        stored_pdf_paths = set(Path(p) for p in self.doc_hashes.keys())
        files_to_process = [p for p in current_pdf_paths if self._needs_update(p)]
        files_to_delete = stored_pdf_paths - current_pdf_paths

        # --- Process Deletions ---
        if files_to_delete:
            self._delete_files_from_db(files_to_delete)

        # --- Process New/Modified Files ---
        if not files_to_process:
            logger.info("No new or modified files to process. Vector store is up to date.")
        else:
            logger.info(f"Found {len(files_to_process)} new/modified files to process.")
            all_new_chunks = []
            max_workers = min(10, multiprocessing.cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(data_processing.extract_and_chunk_with_docling, pdf_path): pdf_path for
                           pdf_path in files_to_process}

                progress_bar = tqdm(as_completed(futures), total=len(files_to_process), desc="Processing PDFs")
                for future in progress_bar:
                    pdf_path = futures[future]
                    progress_bar.set_description(f"Processing {pdf_path.name}")
                    try:
                        doc_chunks = future.result()
                        if doc_chunks:
                            all_new_chunks.extend(doc_chunks)
                            self.doc_hashes[str(pdf_path)] = data_processing.get_file_hash(pdf_path)
                            # IMPORTANT: Don't save hashes here yet. Save only after successful DB write.
                    except Exception as exc:
                        logger.error(f'{pdf_path.name} generated an exception during processing: {exc}', exc_info=True)

            # ### FIX: Add new chunks to the database IN BATCHES with a progress bar. ###
            if all_new_chunks:
                logger.info(f"Adding {len(all_new_chunks)} new chunks to the vector store...")

                # Use a smaller batch size for adding to the DB to provide more frequent progress updates
                db_batch_size = 64
                for i in tqdm(range(0, len(all_new_chunks), db_batch_size), desc="Writing to VectorDB"):
                    batch = all_new_chunks[i:i + db_batch_size]
                    try:
                        self.vectordb.add_documents(documents=batch)
                    except Exception as db_exc:
                        logger.error(f"Failed to add a batch of {len(batch)} chunks to DB: {db_exc}")

                logger.info("Finished adding new chunks.")

        # --- Persist all changes and save hashes ---
        logger.info("Persisting all database changes and saving file hashes...")
        self.vectordb.persist()
        self._save_doc_hashes()  # Save all hashes at the end of a successful sync.
        logger.info("Database sync complete.")
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
                    if str(pdf_path) in self.doc_hashes:
                        del self.doc_hashes[str(pdf_path)]
            except Exception as e:
                logger.error(f"Failed to delete chunks from the database: {e}")
        else:
            logger.warning("Found files marked for deletion, but no corresponding chunks were found in the DB.")

    # The helper _needs_update is still required.
    def _needs_update(self, file_path: Path) -> bool:
        """Checks if a file is new or has been modified since the last run."""
        current_hash = data_processing.get_file_hash(file_path)
        if not current_hash:
            return False
        stored_hash = self.doc_hashes.get(str(file_path))
        return current_hash != stored_hash

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


    def _load_doc_hashes(self) -> Dict[str, str]:
        if self.doc_hashes_file.exists():
            try:
                with open(self.doc_hashes_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_doc_hashes(self):
        with open(self.doc_hashes_file, 'w') as f: json.dump(self.doc_hashes, f, indent=2)

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
        stats = {
            "embedding_model": config.EMBEDDING_MODEL_NAME, "llm_model": config.OPENAI_MODEL_NAME,
            "total_documents": len(self.doc_hashes), "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap, "base_directory": str(self.base_dir),
            "base_directory_exists": self.base_dir.exists()
        }
        if self.vectordb:
            try:
                stats["total_chunks"] = self.vectordb._collection.count()
            except Exception:
                stats["total_chunks"] = "N/A"
        return stats

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
        if not self.retriever:
            yield "Error: RAG system's retriever is not ready."
            return

        yield "Step 1: Analyzing query and generating a hypothetical answer for smarter search (HyDE)..."
        logger.info(f"--- Starting new query: '{question}' ---")

        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert on regulatory documents. Generate a detailed, hypothetical answer to the user's question. This answer should contain the kind of language, data, and specific terms you would expect to find in a real document that answers the question."),
            ("human", "{question}")
        ])
        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        hypothetical_answer = hyde_chain.invoke({"question": question})
        logger.info(f"Generated Hypothetical Answer for embedding: {hypothetical_answer[:200]}...")

        yield "Step 2: Retrieving relevant documents from the corpus..."
        base_retriever = self.vectordb.as_retriever(search_kwargs={"k": 20})
        retrieved_docs = base_retriever.get_relevant_documents(hypothetical_answer)

        if re.search(r'["$]\d', question) or re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', question):
            yield "Literal term detected. Performing additional keyword search..."
            keyword_docs = self.vectordb.similarity_search(question, k=10)
            retrieved_docs.extend(keyword_docs)
            unique_ids = set()
            all_docs = retrieved_docs
            retrieved_docs = []
            for doc in all_docs:
                if doc.metadata['chunk_id'] not in unique_ids:
                    retrieved_docs.append(doc)
                    unique_ids.add(doc.metadata['chunk_id'])

        if not retrieved_docs:
            yield "Could not find relevant documents to answer the question."
            return

        yield "Step 3: Re-ranking documents for precision..."
        reranked_docs = self._rerank_documents(question, retrieved_docs)
        final_docs = reranked_docs[:config.TOP_K_DOCS]
        logger.info(f"Re-ranked documents. Top source: {final_docs[0].metadata['source'] if final_docs else 'None'}")

        yield "Step 4: Synthesizing technical answer..."
        default_no_answer = "The provided context does not contain sufficient information to answer this question."
        part1_answer = default_no_answer
        if self.llm:
            context = self._enhance_context_for_llm(final_docs, question)
            prompt = self.technical_prompt.format(context=context, question=question,
                                                  current_date=datetime.now().strftime("%B %d, %Y"))
            part1_answer_obj = self.llm.invoke(prompt)
            part1_answer = part1_answer_obj.content

        yield "Step 5: Generating a simplified explanation..."
        part2_summary = "A simplified explanation could not be generated."
        if self.llm and part1_answer != default_no_answer:
            prompt = self.layman_prompt.format(technical_answer=part1_answer)
            part2_summary_obj = self.llm.invoke(prompt)
            part2_summary = part2_summary_obj.content

        yield "Step 6: Performing internet search for additional context..."
        part3_search = self._perform_internet_search(question)

        yield "Finalizing the response..."
        final_answer = self._assemble_final_answer(part1_answer, part2_summary, part3_search)

        result_payload = {
            "answer": final_answer,
            "sources": self._process_sources(final_docs),
            "confidence_indicators": self._assess_confidence(final_docs, part1_answer, question)
        }
        yield result_payload

        # Replace the query function and add these new helper methods in rag_core.py

    def _perform_internet_search(self, question: str) -> str:
        """Performs and formats an internet search."""
        try:
            search_query = self._create_search_query(question)
            search_results_str = self.search_tool.run(search_query)
            pattern = re.compile(r"snippet: (.*?),\s*title: (.*?),\s*link: (https?://[^\s,]+)")
            matches = pattern.findall(search_results_str)
            if matches:
                search_results = [{'snippet': s.strip(), 'title': t.strip(), 'link': l.strip()} for s, t, l in
                                  matches]
                return self._format_search_results(search_results)
            return f"<p>Search returned unstructured results.</p>"
        except Exception as e:
            logger.error(f"Internet search failed: {e}")
            return "<p>An error occurred during internet search.</p>"

    def _assemble_final_answer(self, part1, part2, part3) -> str:
        """Assembles the final HTML-formatted answer string."""
        sanitized_part1 = re.sub(r'```[\s\S]*?```', '', part1).replace('`', '')
        sanitized_part2 = re.sub(r'```[\s\S]*?```', '', part2).replace('`', '')
        return f"""
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #dee2e6;">
        <h3 style="border-bottom: 2px solid #0d6efd; padding-bottom: 5px; color: #0d6efd;">Part 1: Direct Answer from Corpus</h3>
        <p style="white-space: pre-wrap;">{sanitized_part1}</p>
    </div>
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #dee2e6;">
        <h3 style="border-bottom: 2px solid #198754; padding-bottom: 5px; color: #198754;">Part 2: Simplified Explanation</h3>
        <p style="white-space: pre-wrap;">{sanitized_part2}</p>
    </div>
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #dee2e6;">
        <h3 style="border-bottom: 2px solid #6c757d; padding-bottom: 5px; color: #6c757d;">Part 3: Additional Context from Web Search</h3>
        <div style="white-space: pre-wrap;">{part3}</div>
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

    def _format_search_results(self, results: List[Dict]) -> str:
        """
        Formats the list of search result dictionaries into a clean, unstyled Markdown string.
        """
        if not results:
            return "No valid search results could be formatted."

        # Use a list to build the parts of the string
        markdown_parts = []
        for i, result in enumerate(results):
            title = result.get('title', 'No Title Provided').strip()
            link = result.get('link', '').strip()
            snippet = result.get('snippet', 'No snippet available.').strip()

            # Create a standard Markdown link: [Title](URL)
            # Only add the link if it's a valid http/https URL
            if link.startswith('http'):
                markdown_parts.append(f"{i + 1}. **[{title}]({link})**")
            else:
                markdown_parts.append(f"{i + 1}. **{title}** (Link not available)")

            # Add the snippet on a new line, formatted as a blockquote
            markdown_parts.append(f"> {snippet}")

        # Join all parts with double newlines for proper spacing in Markdown
        return "\n\n".join(markdown_parts)

    # --- New Private Helper Methods for Search ---
    def _create_search_query(self, question: str) -> str:
        """Creates a search-engine-friendly query from the user's question."""
        # A more advanced version could use an LLM to generate a good search query.
        return f"CPUC regulatory information on: {question}"

    def _parse_date_from_filename(self, pdf_path: Path) -> datetime:
        name = pdf_path.stem.lower()
        decision_match = re.search(r'(?:d|r)\.(\d{2})-(\d{2})-\d{3}', name)
        if decision_match:
            try:
                year_yy, month_mm = decision_match.groups()
                year, month = 2000 + int(year_yy), int(month_mm)
                if 1 <= month <= 12: return datetime(year, month, 1)
            except (ValueError, IndexError):
                pass

        date_match = re.search(
            r'(\d{4})[-_](\d{2})[-_](\d{2})|'r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2}),?\s+(\d{4})',
            name)
        if date_match:
            try:
                if date_match.group(1):
                    year, month, day = map(int, date_match.groups()[0:3])
                    if 1 <= month <= 12: return datetime(year, month, day)
                elif date_match.group(4):
                    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                                 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                    month_str, day, year = date_match.group(4), int(date_match.group(5)), int(date_match.group(6))
                    return datetime(year, month_map[month_str[:3]], day)
            except (ValueError, IndexError, KeyError):
                pass

        return datetime(1900, 1, 1)

    def build_vector_store(self, force_rebuild: bool = False):
        """
        Builds or incrementally updates the vector store in parallel.
        - On first run, processes all PDFs in parallel and builds the DB.
        - On subsequent runs, finds new/modified/deleted PDFs and syncs the DB.
        """
        if force_rebuild and self.db_dir.exists():
            logger.warning("Force rebuild requested. Deleting existing vector store.")
            shutil.rmtree(self.db_dir)
            self.vectordb = None
            self.doc_hashes = {}  # Reset hashes on rebuild

        # --- Load or Initialize VectorDB ---
        if self.db_dir.exists() and any(self.db_dir.iterdir()):
            logger.info("Loading existing vector store...")
            try:
                self.vectordb = Chroma(
                    persist_directory=str(self.db_dir),
                    embedding_function=self.embedding_model
                )
            except Exception as e:
                logger.error(f"Failed to load existing DB, it might be corrupt. Rebuilding. Error: {e}")
                shutil.rmtree(self.db_dir)
                self.vectordb = None
                self.doc_hashes = {}

        if self.vectordb is None:
            logger.info("Initializing new vector store.")
            # We initialize an empty DB. We will add documents to it incrementally.
            self.vectordb = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=str(self.db_dir)
            )

        # --- Sync Files: Find new, modified, and deleted PDFs ---
        current_pdf_paths = {p for p in self.base_dir.rglob("*.pdf")}
        stored_pdf_paths = set(Path(p) for p in self.doc_hashes.keys())

        # Files to add/update are those present on disk whose hash is new or different.
        files_to_process = []
        for pdf_path in current_pdf_paths:
            if self._needs_update(pdf_path):
                files_to_process.append(pdf_path)

        # Files to delete are those in the hash map but no longer on disk.
        files_to_delete = stored_pdf_paths - current_pdf_paths

        # --- Process Deletions ---
        if files_to_delete:
            logger.info(f"Found {len(files_to_delete)} files to delete from vector store.")
            ids_to_delete = []
            for pdf_path in files_to_delete:
                # Find all chunks associated with this file to delete them
                results = self.vectordb.get(where={"source": pdf_path.name})
                ids_to_delete.extend(results['ids'])
                del self.doc_hashes[str(pdf_path)]  # Remove from our hash tracking

            if ids_to_delete:
                logger.info(f"Deleting {len(ids_to_delete)} chunks from the database.")
                self.vectordb.delete(ids=ids_to_delete)

        if not files_to_process:
            logger.info("No new or modified files to process. Vector store is up to date.")
        else:
            logger.info(f"Found {len(files_to_process)} new/modified files to process in parallel.")
            all_new_chunks = []

            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(data_processing.extract_and_chunk_with_docling, pdf_path): pdf_path for
                           pdf_path in files_to_process}

                # ### FIX: Initialize tqdm and update its description in the loop ###
                progress_bar = tqdm(as_completed(futures), total=len(files_to_process), desc="Processing PDFs")

                for future in progress_bar:
                    pdf_path = futures[future]
                    # Update the progress bar to show which file is currently being processed or has just finished.
                    progress_bar.set_description(f"Processing {pdf_path.name}")

                    try:
                        doc_chunks = future.result()
                        if doc_chunks:
                            all_new_chunks.extend(doc_chunks)
                            self.doc_hashes[str(pdf_path)] = data_processing.get_file_hash(pdf_path)
                            # Optionally, log completion here as well
                            logger.info(f"✅ Completed processing for: {pdf_path.name}")
                    except Exception as exc:
                        logger.error(f'{pdf_path.name} generated an exception: {exc}')

            # --- Add new chunks to the database ---
            if all_new_chunks:
                logger.info(f"Adding {len(all_new_chunks)} new chunks to the vector store...")
                # Add the newly processed chunks to the existing DB instance
                self.vectordb.add_documents(documents=all_new_chunks)
                logger.info("Finished adding new chunks.")

        # --- Persist all changes and save hashes ---
        logger.info("Persisting all database changes...")
        self.vectordb.persist()
        self._save_doc_hashes()
        logger.info("Database sync complete.")
        self.setup_qa_pipeline()

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

    def _rank_by_regulatory_relevance(self, documents: List[Document]) -> List[Document]:
        """Rank documents by regulatory relevance, prioritizing decisions over proposals."""
        section_weights = {'order': 1.0, 'finding': 0.9, 'rule': 0.8, 'requirement': 0.7}
        scored_docs = []
        for doc in documents:
            score = 0.0
            doc_name_lower = doc.metadata.get('source', '').lower()
            section_type = doc.metadata.get('section_type', 'unknown')
            chunk_type = doc.metadata.get('chunk_type', 'regular')
            score += section_weights.get(section_type, 0.5)  # Base score
            score += section_weights.get(chunk_type, 0.5)

            ### ENHANCEMENT: Prioritize document types based on filename.
            if any(term in doc_name_lower for term in ["decision", "order", "ruling"]):
                score += 0.4  # High boost for confirmed actions
            if any(term in doc_name_lower for term in ["proposal", "draft"]):
                score -= 0.2  # Penalize preliminary documents

            doc.metadata['relevance_score'] = round(score, 2)
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]

    def _identify_regulatory_sections(self, text: str) -> List[Dict]:
        sections = []
        section_patterns = {
            'order': r'((?:IT IS ORDERED|O R D E R|O R D E R S)[\s\S]+)',
            'finding': r'((?:FINDINGS OF FACT|F I N D I N G S)[\s\S]+?(?=(?:IT IS ORDERED|O R D E R|CONCLUSION)))',
        }
        for sec_type, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match: sections.append({'type': sec_type, 'content': match.group(1).strip()})
        return sections

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
            bool(re.search(r'\[Source:.*?Page: \d+\]', answer))
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

    def _preprocess_query(self, question: str) -> str:
        expansions = {"deadline": "deadline due date filing date", "requirement": "requirement must shall obligation",
                      "cost": "cost rate tariff fee charge"}
        processed_q = question.lower()
        for term, terms in expansions.items():
            if term in processed_q: return question + " " + terms
        return question

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

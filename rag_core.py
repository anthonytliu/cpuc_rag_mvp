# 📁 rag_core.py
# The core RAG system logic.

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchResults
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

    def query(self, question: str) -> Dict:
        if not self.retriever:
            logger.error("Query attempted but retriever is not initialized.")
            return {"answer": "Error: RAG system's retriever is not ready.", "sources": [], "confidence_indicators": {}}
        logger.info(f"--- Starting new query: '{question}' ---")
        logger.info("Step 1: Performing RAG retrieval for technical answer.")
        relevant_docs = self._hybrid_retrieval(question)
        logger.info(f"Retrieved {len(relevant_docs)} documents for context.")
        default_no_answer = "The provided context does not contain sufficient information to answer this question."
        part1_answer = default_no_answer
        if self.llm and relevant_docs:
            context = self._enhance_context_for_llm(relevant_docs, question)
            formatted_technical_prompt = self.technical_prompt.format(context=context, question=question,
                                                                      current_date=datetime.now().strftime("%B %d, %Y"))
            try:
                # ### FIX: The invoke method returns an AIMessage object. Extract the string content. ###
                response_message = self.llm.invoke(formatted_technical_prompt)
                part1_answer = response_message.content  # <-- This is the fix

                logger.info(f"Successfully generated Part 1 (technical answer). Length: {len(part1_answer)} chars.")
            except Exception as e:
                logger.error(f"Error invoking LLM for Part 1: {e}", exc_info=True)
                part1_answer = "An error occurred while generating the technical answer from the corpus."
        else:
            logger.warning("No relevant documents found or LLM not available for Part 1.")

        logger.info("Step 2: Generating layman's summary.")
        part2_summary = "A simplified explanation could not be generated as no detailed technical answer was available."
        if self.llm and part1_answer != default_no_answer:
            formatted_layman_prompt = self.layman_prompt.format(technical_answer=part1_answer)
            try:
                # ### FIX: Apply the same fix here for the second LLM call. ###
                response_summary_message = self.llm.invoke(formatted_layman_prompt)
                part2_summary = response_summary_message.content  # <-- This is the fix

                logger.info("Successfully generated Part 2 (layman's summary).")
            except Exception as e:
                logger.error(f"Error invoking LLM for Part 2: {e}", exc_info=True)
                part2_summary = "An error occurred while simplifying the technical answer."
        else:
            logger.warning("Skipping layman's summary because Part 1 did not produce a valid answer.")

        # --- PART 3: Perform an internet search for additional context ---
        # ... (The rest of the query method is correct and does not need changes)
        logger.info("Step 3: Performing internet search.")
        part3_search = "Internet search did not return relevant results."
        try:
            search_query = self._create_search_query(question)
            logger.info(f"Using search query: '{search_query}'")
            search_results_str = self.search_tool.run(search_query)
            logger.info(f"Raw search tool output: {search_results_str[:300]}...")
            pattern = re.compile(r"snippet: (.*?),\s*title: (.*?),\s*link: (https?://[^\s,]+)")
            matches = pattern.findall(search_results_str)
            if matches:
                search_results = [{'snippet': snippet.strip(), 'title': title.strip(), 'link': link.strip()} for
                                  snippet, title, link in matches]
                part3_search = self._format_search_results(search_results)
                logger.info(f"Successfully parsed {len(matches)} search results using enhanced regex.")
            else:
                logger.warning("Could not parse search results with enhanced regex. Displaying raw text.")
                part3_search = f"<p>Search returned results in an unrecognized format. Raw output:</p><p><code>{search_results_str}</code></p>"
        except Exception as e:
            logger.error(f"Internet search tool failed entirely: {e}", exc_info=True)
            part3_search = "<p>An error occurred during the internet search.</p>"
        sanitized_part1 = re.sub(r'```[\s\S]*?```', '', part1_answer).replace('`', '')
        sanitized_part2 = re.sub(r'```[\s\S]*?```', '', part2_summary).replace('`', '')
        final_answer = f"""
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #dee2e6;">
        <h3 style="border-bottom: 2px solid #0d6efd; padding-bottom: 5px; color: #0d6efd;">Part 1: Direct Answer from Corpus</h3>
        <p style="white-space: pre-wrap;">{sanitized_part1}</p>
        <br>
        <h3 style="border-bottom: 2px solid #198754; padding-bottom: 5px; color: #198754;">Part 2: Simplified Explanation</h3>
        <p style="white-space: pre-wrap;">{sanitized_part2}</p>
        <br>
        <h3 style="border-bottom: 2px solid #6c757d; padding-bottom: 5px; color: #6c757d;">Part 3: Additional Context from Web Search</h3>
        <div style="white-space: pre-wrap;">{part3_search}</div>
    </div>
    """
        sources = self._process_sources(relevant_docs)
        confidence = self._assess_confidence(relevant_docs, part1_answer, question)
        self.conversation_memory.add_query_context(question, final_answer, sources,
                                                   self.conversation_memory.extract_entities(part1_answer))
        return {"answer": final_answer, "sources": sources, "confidence_indicators": confidence}

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
        """Creates a search-engine friendly query from the user's question."""
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
        if force_rebuild and self.db_dir.exists():
            logger.warning("Force rebuild requested. Deleting existing vector store.")
            shutil.rmtree(self.db_dir)
            self.vectordb = None

        if not force_rebuild and self.vectordb is None and self.db_dir.exists() and any(self.db_dir.iterdir()):
            try:
                logger.info("Attempting to load existing vector store...")
                self.vectordb = Chroma(persist_directory=str(self.db_dir), embedding_function=self.embedding_model)
                logger.info("Vector store loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load database: {e}. Rebuilding...")
                shutil.rmtree(self.db_dir)
                self.vectordb = None

        if self.vectordb is None:
            logger.info("Building new vector store...")
            self.db_dir.mkdir(exist_ok=True)
            pdf_files = sorted(list(self.base_dir.rglob("*.pdf")), key=self._parse_date_from_filename, reverse=True)

            all_docs = []
            for pdf_path in pdf_files:
                if self._needs_update(pdf_path):
                    docs = data_processing.extract_text_from_pdf(pdf_path)
                    all_docs.extend(docs)
                    self.doc_hashes[str(pdf_path)] = data_processing.get_file_hash(pdf_path)

            if not all_docs:
                logger.warning("No new documents to process. Existing DB (if any) will be used.")
            else:
                chunked_docs = self._create_hierarchical_chunks(all_docs)
                if chunked_docs:
                    logger.info(f"Generating embeddings for {len(chunked_docs)} chunks. This will take a while...")

                    ### FIX: Replace from_documents with a manual loop + progress bar ###
                    # Create an empty DB first
                    self.vectordb = Chroma(
                        embedding_function=self.embedding_model,
                        persist_directory=str(self.db_dir)
                    )

                    # Add documents in batches with a progress bar
                    batch_size = 128  # A good batch size for local processing
                    for i in tqdm(range(0, len(chunked_docs), batch_size), desc="Embedding Chunks"):
                        batch = chunked_docs[i:i + batch_size]
                        self.vectordb.add_documents(documents=batch)

                    self.vectordb.persist()
                    self._save_doc_hashes()
                    logger.info("New vector store built and persisted.")

        self.setup_qa_pipeline()

    def setup_qa_pipeline(self):
        if self.vectordb is None:
            logger.error("Vector DB is not available. Cannot setup QA pipeline.")
            return
        self.retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50}
        )
        logger.info("QA pipeline and retriever are ready.")

    def _rank_by_regulatory_relevance(self, documents: List[Document]) -> List[Document]:
        scored_docs = []
        for doc in documents:
            score = 0.0
            doc_name_lower = doc.metadata.get('source', '').lower()
            if any(term in doc_name_lower for term in ["decision", "order", "ruling"]):
                score += 0.5
            elif any(term in doc_name_lower for term in ["resolution"]):
                score += 0.4
            elif any(term in doc_name_lower for term in ["proposal", "draft"]):
                score -= 0.2
            elif any(term in doc_name_lower for term in ["testimony", "comment"]):
                score -= 0.3
            try:
                mod_time = doc.metadata.get('last_modified')
                if mod_time:
                    doc_date = datetime.fromisoformat(mod_time)
                    days_old = (datetime.now() - doc_date).days
                    if days_old < 90:
                        score += 0.3
                    elif days_old < 365:
                        score += 0.1
            except ValueError:
                pass
            doc.metadata['relevance_score'] = round(score, 2)
            scored_docs.append((doc, score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]

    def _hybrid_retrieval(self, query: str) -> List[Document]:
        enhanced_question = self._preprocess_query(query)
        retrieved_docs = self.retriever.get_relevant_documents(enhanced_question)
        ranked_docs = self._rank_by_regulatory_relevance(retrieved_docs)
        unique_docs_map = {}
        for doc in ranked_docs:
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if content_hash not in unique_docs_map: unique_docs_map[content_hash] = doc
        return list(unique_docs_map.values())[:config.TOP_K_DOCS]

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

    def _create_hierarchical_chunks(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            sections = self._identify_regulatory_sections(doc.page_content)
            if sections:
                for section in sections:
                    if len(section['content']) > self.chunk_size:
                        section_doc = Document(page_content=section['content'],
                                               metadata={**doc.metadata, 'section_type': section['type']})
                        child_chunks = data_processing.chunk_documents([section_doc], self.chunk_size,
                                                                       self.chunk_overlap)
                        all_chunks.extend(child_chunks)
                    else:
                        parent_chunk = Document(
                            page_content=section['content'],
                            metadata={**doc.metadata, 'chunk_type': 'parent', 'section_type': section['type']}
                        )
                        all_chunks.append(parent_chunk)
            else:
                all_chunks.extend(data_processing.chunk_documents([doc], self.chunk_size, self.chunk_overlap))
        logger.info(f"Hierarchical chunking complete. Total chunks created: {len(all_chunks)}")
        return all_chunks

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

    def _needs_update(self, file_path: Path) -> bool:
        current_hash = data_processing.get_file_hash(file_path)
        return current_hash != self.doc_hashes.get(str(file_path))

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

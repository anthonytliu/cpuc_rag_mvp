# 📁 rag_core.py
# The core RAG system logic.
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

# Internal imports from our structured project
import config
import data_processing
import models
import utils
from memory import ConversationMemory
from models import setup_local_models

logger = logging.getLogger(__name__)


class CPUCRAGSystem:
    def __init__(self):
        # Load from config
        self.base_dir = config.BASE_PDF_DIR
        self.db_dir = config.DB_DIR
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.llm_model = config.LLM_MODEL

        # Initialize models
        self.embedding_model = models.get_embedding_model()
        self.llm = models.get_llm()

        # Initialize components
        self.conversation_memory = ConversationMemory()
        self.similarity_retriever = None
        self.semantic_retriever = None
        self.embeddings = None
        self.vectordb = None
        self.retriever = None
        self.doc_hashes_file = self.db_dir / "document_hashes.json"
        self.doc_hashes = self._load_doc_hashes()

        self.prompt = PromptTemplate(
            template=config.ACCURACY_PROMPT_TEMPLATE,
            input_variables=["context", "question", "current_date"]
        )

        # Debug: Print paths and existence
        logger.info(f"Base directory: {self.base_dir.absolute()}")
        logger.info(f"Base directory exists: {self.base_dir.exists()}")
        logger.info(f"DB directory: {self.db_dir.absolute()}")

        # Create directories if they don't exist
        self.db_dir.mkdir(exist_ok=True)

        # FIXED: Check if base directory exists and has PDFs
        if not self.base_dir.exists():
            logger.error(f"Base directory does not exist: {self.base_dir}")
            raise FileNotFoundError(f"Base directory does not exist: {self.base_dir}")

        # Count PDFs in base directory
        pdf_files = list(self.base_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in base directory")

        if len(pdf_files) == 0:
            logger.warning("No PDF files found in base directory!")
            # List all files in directory for debugging
            all_files = list(self.base_dir.iterdir())
            logger.info(f"All files in directory: {[f.name for f in all_files]}")

        setup_local_models(self)

    def build_vector_store(self, force_rebuild: bool = False) -> Chroma:
        """Build or update vector store with all proceedings - FIXED"""
        if not force_rebuild and self.vectordb is not None:
            logger.info("Vector store already exists, skipping build")
            return self.vectordb

        all_documents = []

        # FIXED: Handle a single directory case
        if self.base_dir.is_dir():
            # Check if base_dir contains PDFs directly
            pdf_files = list(self.base_dir.glob("*.pdf"))
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} PDFs in base directory")
                documents = data_processing.process_proceeding(self, self.base_dir)
                all_documents.extend(documents)
            else:
                # Check for subdirectories
                logger.info("No PDFs in base directory, checking subdirectories...")
                for proceeding_dir in self.base_dir.iterdir():
                    if proceeding_dir.is_dir():
                        logger.info(f"Processing subdirectory: {proceeding_dir.name}")
                        documents = data_processing.process_proceeding(self, proceeding_dir)
                        all_documents.extend(documents)

        if not all_documents:
            logger.error("No documents found to process!")
            logger.info("Available files in base directory:")
            for item in self.base_dir.iterdir():
                logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            return None

        # Chunk all documents
        logger.info(f"Chunking {len(all_documents)} documents...")
        chunked_docs = data_processing._create_hierarchical_chunks(self, all_documents)

        if not chunked_docs:
            logger.error("No chunks created!")
            return None

        # Create vector store
        logger.info(f"Creating vector store with {len(chunked_docs)} chunks...")

        # Remove the existing database if force rebuilds
        if force_rebuild and self.db_dir.exists():
            import shutil
            shutil.rmtree(self.db_dir)
            self.db_dir.mkdir(exist_ok=True)

        try:
            # Process in smaller batches for stability
            batch_size = 100

            if len(chunked_docs) <= batch_size:
                # Small dataset - process all at once
                self.vectordb = Chroma.from_documents(
                    documents=chunked_docs,
                    embedding=self.embeddings,
                    persist_directory=str(self.db_dir),
                    collection_name="cpuc_documents"
                )
            else:
                # Large dataset - process in batches
                logger.info(f"Processing in batches of {batch_size}...")

                # Create initial vector store with first batch
                first_batch = chunked_docs[:batch_size]
                self.vectordb = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=str(self.db_dir),
                    collection_name="cpuc_documents"
                )

                # Add remaining batches
                for i in range(batch_size, len(chunked_docs), batch_size):
                    batch = chunked_docs[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(chunked_docs) - 1) // batch_size + 1

                    logger.info(f"Processing batch {batch_num}/{total_batches}")

                    # Add to existing vector store
                    self.vectordb.add_documents(batch)

                    # Persist after each batch
                    self.vectordb.persist()

            self.vectordb.persist()
            self._save_doc_hashes()

            logger.info("Vector store created successfully!")
            return self.vectordb

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None

    def setup_qa_pipeline(self):
        """Setup QA pipeline with hybrid retrieval"""
        if self.vectordb is None:
            try:
                self.vectordb = Chroma(
                    persist_directory=str(self.db_dir),
                    embedding_function=self.embeddings,
                    collection_name="cpuc_documents"
                )
                logger.info("Loaded existing vector store")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                return

        # Setup multiple retrievers for hybrid approach
        try:
            # Primary semantic retriever (more results)
            self.semantic_retriever = self.vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 12, "fetch_k": 24, "lambda_mult": 0.3}  # Increased retrieval
            )

            # Similarity retriever for backup
            self.similarity_retriever = self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )

            # Use semantic as primary
            self.retriever = self.semantic_retriever

            logger.info("Enhanced QA pipeline setup complete")
        except Exception as e:
            logger.error(f"Error setting up retriever: {e}")

    def _hybrid_retrieval(self, query: str) -> List[Document]:
        """Enhanced hybrid retrieval with regulatory context awareness"""
        try:
            # 1. Standard semantic retrieval
            semantic_docs = self.semantic_retriever.get_relevant_documents(query)

            # 2. Multi-pass retrieval for cross-references
            cross_ref_docs = self._multi_pass_retrieval(query)

            # 3. Get parent sections for any child chunks
            parent_docs = self._get_parent_sections(semantic_docs)

            # 4. Combine and deduplicate
            all_docs = semantic_docs + cross_ref_docs + parent_docs
            seen_content = set()
            unique_docs = []

            for doc in all_docs:
                content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)

            # 5. Rank by regulatory relevance
            ranked_docs = utils._rank_by_regulatory_relevance(unique_docs, query)

            return ranked_docs[:15]

        except Exception as e:
            logger.error(f"Enhanced hybrid retrieval error: {e}")
            return self.retriever.get_relevant_documents(query)

    def _get_parent_sections(self, documents: List[Document]) -> List[Document]:
        """Get parent sections for child chunks"""
        parent_docs = []

        for doc in documents:
            if doc.metadata.get('chunk_type') == 'child':
                parent_title = doc.metadata.get('parent_title', '')
                parent_section = doc.metadata.get('parent_section', '')

                if parent_title:
                    # Search for a parent document
                    parent_query = f"section_title:{parent_title} OR section_type:{parent_section}"
                    try:
                        parent_results = self.vectordb.similarity_search(parent_query, k=2)
                        for parent in parent_results:
                            if parent.metadata.get('chunk_type') == 'parent':
                                parent_docs.append(parent)
                    except:
                        pass

        return parent_docs

    def _multi_pass_retrieval(self, query: str) -> List[Document]:
        """Enhanced multi-pass retrieval with cross-document analysis"""
        # First pass - standard retrieval
        initial_docs = self.retriever.get_relevant_documents(query)

        # Second pass - find cross-referenced documents
        cross_refs = utils._find_cross_references(initial_docs)

        additional_docs = []
        for source, refs in cross_refs.items():
            for ref in refs:
                # Search for documents containing these references
                ref_query = f"proceeding {ref} OR docket {ref} OR case {ref}"
                ref_docs = self.retriever.get_relevant_documents(ref_query)
                additional_docs.extend(ref_docs[:2])  # Limit to avoid explosion

        # Combine and deduplicate
        all_docs = initial_docs + additional_docs
        seen_content = set()
        unique_docs = []

        for doc in all_docs:
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs[:15]

    def query(self, question: str) -> Dict:
        """Enhanced query with better retrieval and context"""
        if self.retriever is None:
            self.setup_qa_pipeline()

        if self.retriever is None:
            return {"answer": "Error: System not properly initialized.", "sources": [], "confidence_indicators": {}}

        try:
            # Enhanced query preprocessing
            enhanced_question = _preprocess_query(question)
            logger.info(f"Enhanced query: {enhanced_question[:100]}...")

            # Use hybrid retrieval if available
            if hasattr(self, '_hybrid_retrieval'):
                relevant_docs = self._hybrid_retrieval(enhanced_question)
            else:
                relevant_docs = self.retriever.get_relevant_documents(enhanced_question)

            logger.info(f"Retrieved {len(relevant_docs)} documents")

            # Enhanced context creation
            context = _enhance_context_for_llm(relevant_docs, question)

            # Add conversation context
            conversation_context = self.conversation_memory.get_relevant_history(question)
            if conversation_context:
                context = f"CONVERSATION CONTEXT:\n{conversation_context}\n\n{context}"

            # Generate answer with enhanced prompt
            if self.llm is not None:
                try:
                    current_date = datetime.now().strftime("%B %d, %Y")

                    # Enhanced prompt with better instructions
                    enhanced_prompt = f"""You are a CPUC regulatory analyst. Current date: {current_date}

                CONTEXT: {context}

                QUESTION: {question}

                INSTRUCTIONS: Provide a comprehensive answer that:
                1. Directly answers the question with specific details
                2. Cites exact sources (document name and page number)
                3. Notes any time-sensitive information relative to {current_date}
                4. Identifies any regulatory obligations, deadlines, or requirements
                5. Explains the regulatory significance and implications
                6. Clearly states if information is incomplete or if additional research is needed

                Focus on being thorough and analytical - a human analyst would catch nuances, cross-references, and implications that might not be immediately obvious.

                ANSWER:"""

                    answer = self.llm(enhanced_prompt)

                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    answer = utils._create_fallback_answer(context, question)
            else:
                answer = utils._create_fallback_answer(context, question)

            # Extract entities from answer for memory
            entities = self.conversation_memory.extract_entities(answer)

            # Enhanced result processing
            processed_result = {
                "answer": answer,
                "sources": _process_sources(relevant_docs),
                "confidence_indicators": utils._assess_confidence(self, relevant_docs, answer, question),
                "query_enhancement": {
                    "original_query": question,
                    "enhanced_query": enhanced_question,
                    "documents_retrieved": len(relevant_docs)
                }
            }

            # Store in conversation memory
            self.conversation_memory.add_query_context(question, answer, processed_result["sources"], entities)

            return processed_result

        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                "answer": f"Error occurred during query: {e}",
                "sources": [],
                "confidence_indicators": {}
            }

    def _load_doc_hashes(self) -> Dict[str, str]:
        """Load document hashes for change detection"""
        if self.doc_hashes_file.exists():
            try:
                with open(self.doc_hashes_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading doc hashes: {e}")
                return {}
        return {}

    def _save_doc_hashes(self):
        """Save document hashes"""
        try:
            with open(self.doc_hashes_file, 'w') as f:
                json.dump(self.doc_hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving doc hashes: {e}")

    def _needs_update(self, file_path: Path) -> bool:
        """Check if the file needs to be reprocessed"""
        current_hash = data_processing._get_file_hash(file_path)
        if not current_hash:
            return False

        stored_hash = self.doc_hashes.get(str(file_path), "")
        needs_update = current_hash != stored_hash

        if needs_update:
            logger.info(f"File needs update: {file_path.name}")

        return needs_update

    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model if self.llm else "None (Retrieval Only)",
            "device": "MPS (M4 MacBook)",
            "total_documents": len(self.doc_hashes),
            "database_location": str(self.db_dir),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "base_directory": str(self.base_dir.absolute()),
            "base_directory_exists": self.base_dir.exists()
        }

        if self.vectordb:
            try:
                collection = self.vectordb._collection
                stats["total_chunks"] = collection.count()
            except:
                stats["total_chunks"] = "Unknown"

        return stats


def _process_sources(documents: List[Document]) -> List[Dict]:
    """Enhanced source processing with better metadata"""
    sources = []

    for i, doc in enumerate(documents):
        source_info = {
            "rank": i + 1,
            "document": doc.metadata.get("source", "Unknown"),
            "proceeding": doc.metadata.get("proceeding", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "excerpt": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
            "last_modified": doc.metadata.get("last_modified", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
            "file_path": doc.metadata.get("file_path", "Unknown")
        }
        sources.append(source_info)

    return sources


def _enhance_context_for_llm(relevant_docs: List[Document], question: str) -> str:
    """Enhanced context creation with better organization and analysis"""

    # Sort documents by relevance score if available, otherwise by source
    sorted_docs = sorted(relevant_docs, key=lambda x: (
        x.metadata.get('source', 'Z'),
        x.metadata.get('page', 999)
    ))

    # Group by document and analyze content
    doc_groups = {}
    for doc in sorted_docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        key = f"{source}_p{page}"

        if key not in doc_groups:
            doc_groups[key] = {
                'docs': [],
                'combined_content': '',
                'relevance_score': 0
            }

        doc_groups[key]['docs'].append(doc)
        doc_groups[key]['combined_content'] += ' ' + doc.page_content

        # Simple relevance scoring based on question keywords
        question_words = set(question.lower().split())
        content_words = set(doc.page_content.lower().split())
        overlap = len(question_words.intersection(content_words))
        doc_groups[key]['relevance_score'] += overlap

    # Sort groups by relevance (handle both dict and list structures)
    sorted_groups = sorted(doc_groups.items(),
                           key=lambda x: x[1].get('relevance_score', 0) if isinstance(x[1], dict) else 0,
                           reverse=True)

    # Create enhanced context
    context_parts = []
    current_date = datetime.now().strftime("%B %d, %Y")

    # Add context header with analysis guidance
    context_header = f"""REGULATORY ANALYSIS CONTEXT (Current Date: {current_date})
        Question: {question}

        INSTRUCTIONS: The following document excerpts contain information relevant to the question. 
        Pay special attention to:
        - Specific deadlines, dates, and time-sensitive requirements
        - Regulatory obligations and compliance requirements  
        - Cross-references to other proceedings or documents
        - Any conditional language ("if", "unless", "provided that")
        - Effective dates and implementation timelines

        DOCUMENT EXCERPTS:
        """

    # Process each document group
    for i, (key, group_data) in enumerate(sorted_groups[:8]):  # Limit to top 8 most relevant
        source_info = key.replace('_p', ' (Page ')
        if not source_info.endswith('Unknown)'):
            source_info += ')'

        # Enhanced content processing
        combined_content = group_data['combined_content'].strip()

        # Add date contextualization
        enhanced_content = utils._extract_and_enhance_dates(combined_content)

        # Add regulatory term highlighting
        enhanced_content = utils._highlight_regulatory_terms(enhanced_content, question)

        # Truncate if too long but preserve important parts
        if len(enhanced_content) > 2000:
            # Try to keep content that matches question keywords
            question_words = question.lower().split()
            sentences = enhanced_content.split('. ')

            # Score sentences by keyword overlap
            scored_sentences = []
            for sentence in sentences:
                score = sum(1 for word in question_words if word in sentence.lower())
                scored_sentences.append((sentence, score))

            # Keep the highest scoring sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            kept_sentences = [s[0] for s in scored_sentences[:10]]  # Top 10 sentences
            enhanced_content = '. '.join(kept_sentences) + '...'

        context_parts.append(f"[{i + 1}] {source_info}:\n{enhanced_content}")

    return context_header + "\n\n".join(context_parts)


def _preprocess_query(question: str) -> str:
    """Enhanced query preprocessing with regulatory domain knowledge"""

    # Regulatory domain expansions
    regulatory_expansions = {
        "deadline": ["deadline", "due date", "filing date", "submission date", "time limit", "expiration"],
        "requirement": ["requirement", "must", "shall", "obligation", "mandate", "necessary", "required"],
        "process": ["process", "procedure", "steps", "method", "timeline", "workflow", "protocol"],
        "decision": ["decision", "ruling", "order", "determination", "finding", "conclusion", "resolution"],
        "compliance": ["compliance", "comply", "adherence", "conformity", "meet requirements"],
        "filing": ["filing", "submit", "file", "lodge", "present", "submission"],
        "comment": ["comment", "public comment", "stakeholder input", "feedback", "response"],
        "effective": ["effective", "implementation", "takes effect", "becomes effective", "in force"],
        "proceeding": ["proceeding", "docket", "case", "matter", "rulemaking", "investigation"],
        "party": ["party", "participant", "intervenor", "respondent", "applicant", "petitioner"],
        "rate": ["rate", "tariff", "charge", "fee", "cost", "price", "billing"],
        "utility": ["utility", "company", "corporation", "provider", "service provider"],
        "commission": ["commission", "CPUC", "authority", "regulator", "agency"],
        "hearing": ["hearing", "proceeding", "session", "meeting", "conference"],
        "notice": ["notice", "notification", "announcement", "advisory", "alert"],
        "application": ["application", "petition", "request", "proposal", "filing"],
        "approval": ["approval", "authorization", "permit", "license", "consent"],
        "review": ["review", "examination", "assessment", "evaluation", "analysis"],
        "modification": ["modification", "amendment", "change", "revision", "update"],
        "suspension": ["suspension", "halt", "stop", "pause", "discontinue"],
        "violation": ["violation", "breach", "non-compliance", "infringement", "contravention"]
    }

    # Legal/regulatory phrases
    legal_phrases = {
        "burden of proof": ["burden of proof", "evidentiary standard", "proof required"],
        "due process": ["due process", "procedural rights", "fair hearing"],
        "good cause": ["good cause", "just cause", "sufficient reason"],
        "public interest": ["public interest", "public benefit", "public welfare"],
        "just and reasonable": ["just and reasonable", "fair and reasonable", "appropriate"]
    }

    original_query = question.lower()
    expanded_terms = []

    # Add regulatory term expansions
    for base_term, expansions in regulatory_expansions.items():
        if any(term in original_query for term in expansions):
            expanded_terms.extend(expansions)

    # Add legal phrase expansions
    for phrase, expansions in legal_phrases.items():
        if phrase in original_query:
            expanded_terms.extend(expansions)

    # Add temporal context terms
    temporal_indicators = ["when", "deadline", "date", "time", "schedule", "expire", "effective", "due", "until"]
    if any(indicator in original_query for indicator in temporal_indicators):
        expanded_terms.extend(
            ["date", "deadline", "timeline", "schedule", "effective date", "expiration", "due date"])

    # Add proceeding context
    expanded_terms.extend(["CPUC", "proceeding", "regulatory", "commission"])

    # Create enhanced query
    enhanced_query = question
    if expanded_terms:
        # Add unique terms only
        unique_terms = list(set(expanded_terms))
        enhanced_query += " " + " ".join(unique_terms)

    return enhanced_query

#!/usr/bin/env python3
"""
Professional RAG System - Complete Implementation (Fully Local)
Advanced document analysis with language model integration - NO INTERNET REQUIRED
"""

import argparse
import os
import sys
import time
import json
import re
import sqlite3
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Core ML and NLP
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
from peft import PeftModel
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Document processing
import PyPDF2
import docx
import pandas as pd

try:
    from pptx import Presentation
except ImportError:
    Presentation = None
try:
    import eml_parser
except ImportError:
    eml_parser = None
try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None


@dataclass
class DocumentChunk:
    """Structured document chunk with metadata"""
    content: str
    source: str
    chunk_id: int
    file_type: str
    page_number: Optional[int] = None
    timestamp: Optional[str] = None
    hash_id: Optional[str] = None
    metadata: Optional[Dict] = None


class OutputFormatter:
    """Advanced output formatting with multiple styles"""

    def __init__(self, professional=False, no_emojis=False, quiet=False):
        self.professional = professional
        self.no_emojis = no_emojis
        self.quiet = quiet
        self.verbose = not quiet

    def format_message(self, message_type: str, message: str) -> str:
        """Format messages based on style configuration"""
        if self.quiet and message_type in ['info', 'debug']:
            return ""

        if self.professional:
            prefixes = {
                'info': '[INFO]',
                'success': '[SUCCESS]',
                'error': '[ERROR]',
                'warning': '[WARNING]',
                'search': '[SEARCHING]',
                'processing': '[PROCESSING]',
                'ingesting': '[INGESTING]',
                'loading': '[LOADING]',
                'saving': '[SAVING]'
            }
            return f"{prefixes.get(message_type, '[INFO]')} {message}"

        elif self.no_emojis:
            prefixes = {
                'info': 'INFO:',
                'success': 'SUCCESS:',
                'error': 'ERROR:',
                'warning': 'WARNING:',
                'search': 'SEARCHING:',
                'processing': 'PROCESSING:',
                'ingesting': 'INGESTING:',
                'loading': 'LOADING:',
                'saving': 'SAVING:'
            }
            return f"{prefixes.get(message_type, 'INFO:')} {message}"

        else:
            prefixes = {
                'info': 'ℹ️',
                'success': '✅',
                'error': '❌',
                'warning': '⚠️',
                'search': '🔍',
                'processing': '⚙️',
                'ingesting': '📥',
                'loading': '📂',
                'saving': '💾'
            }
            return f"{prefixes.get(message_type, 'ℹ️')} {message}"

    def print(self, message_type: str, message: str):
        """Print formatted message"""
        formatted = self.format_message(message_type, message)
        if formatted:
            print(formatted)

    def banner(self, title: str, info: Dict[str, Any]):
        """Print system banner"""
        if self.quiet:
            return

        if self.professional:
            print("=" * 80)
            print(f"{title.upper()}")
            print("Advanced Document Analysis with Language Model Integration (OFFLINE)")
            print("=" * 80)
            for key, value in info.items():
                print(f"{key}: {value}")
            print()
            print("Available Commands:")
            print("  ingest <path>           Ingest documents from directory")
            print("  search <query>          Search knowledge base")
            print("  debug <query>           Debug search results")
            print("  test-summary            Test summarization capability")
            print("  rebuild-indices         Rebuild search indices")
            print("  stats                   Show system statistics")
            print("  formats                 Show supported file formats")
            print("  cluster                 Analyze document clusters")
            print("  export <format>         Export knowledge base")
            print("  clear                   Clear knowledge base")
            print("  quit/exit               Exit system")
            print("=" * 80)
        else:
            emoji = "" if self.no_emojis else "🎭 "
            print(f"\n{emoji}{title} (OFFLINE MODE)")
            if not self.no_emojis:
                print("📚 Commands: ingest, search, debug, stats, cluster, quit")


class AdvancedRAGDatabase:
    """Sophisticated RAG database with multiple search methods - FULLY LOCAL"""

    def __init__(self, output_formatter: OutputFormatter, db_path: str = "professional_rag.db"):
        self.output = output_formatter
        self.db_path = db_path
        self.chunks: List[DocumentChunk] = []
        self.knowledge_base: Dict[str, Dict] = defaultdict(dict)

        # Search engines
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = None

        # Semantic search - LOCAL ONLY
        self.embedding_model = None
        self.faiss_index = None
        self.embeddings = None

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

            # Create tables
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash_id TEXT UNIQUE,
                    content TEXT,
                    source TEXT,
                    chunk_id INTEGER,
                    file_type TEXT,
                    page_number INTEGER,
                    timestamp TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    key TEXT,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(category, key)
                )
            ''')

            self.conn.commit()

        except Exception as e:
            self.output.print('error', f"Database initialization error: {e}")

    def initialize_embeddings(self):
        """Initialize semantic search embeddings - LOCAL ONLY"""
        try:
            self.output.print('loading', "Loading local embedding model")

            # Check for local sentence transformer model
            local_embedding_path = "./sentence-transformer-model"
            if os.path.exists(local_embedding_path):
                self.embedding_model = SentenceTransformer(local_embedding_path)
                self.output.print('success', "Local embedding model loaded successfully")
            else:
                self.output.print('error', f"Local embedding model not found at {local_embedding_path}")
                self.output.print('info',
                                  "Please ensure sentence transformer model is downloaded to ./sentence-transformer-model/")
                self.output.print('warning', "Running without semantic search capabilities")

        except Exception as e:
            self.output.print('error', f"Failed to load local embedding model: {e}")
            self.output.print('warning', "Running without semantic search capabilities")

    def load_existing_data(self):
        """Load existing chunks from database"""
        try:
            self.cursor.execute('SELECT COUNT(*) FROM chunks')
            count = self.cursor.fetchone()[0]

            if count > 0:
                self.output.print('loading', f"Loading {count} existing documents")

                self.cursor.execute('''
                    SELECT hash_id, content, source, chunk_id, file_type, 
                           page_number, timestamp, metadata 
                    FROM chunks
                ''')

                rows = self.cursor.fetchall()
                self.chunks = []

                for row in rows:
                    metadata = json.loads(row[7]) if row[7] else {}
                    chunk = DocumentChunk(
                        content=row[1],
                        source=row[2],
                        chunk_id=row[3],
                        file_type=row[4],
                        page_number=row[5],
                        timestamp=row[6],
                        hash_id=row[0],
                        metadata=metadata
                    )
                    self.chunks.append(chunk)

                # Load knowledge base
                self.cursor.execute('SELECT category, key, value FROM knowledge')
                for category, key, value in self.cursor.fetchall():
                    self.knowledge_base[category][key] = value

                # Rebuild indices
                self._rebuild_search_indices()

                self.output.print('success', f"Loaded existing model with {len(self.chunks)} documents")
            else:
                self.output.print('info', "No existing documents found")

        except Exception as e:
            self.output.print('error', f"Error loading existing data: {e}")

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add new chunks to the database"""
        if not chunks:
            return

        new_chunks = []
        duplicate_count = 0

        for chunk in chunks:
            try:
                # Insert into database
                self.cursor.execute('''
                    INSERT OR IGNORE INTO chunks 
                    (hash_id, content, source, chunk_id, file_type, page_number, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk.hash_id,
                    chunk.content,
                    chunk.source,
                    chunk.chunk_id,
                    chunk.file_type,
                    chunk.page_number,
                    chunk.timestamp,
                    json.dumps(chunk.metadata) if chunk.metadata else None
                ))

                if self.cursor.rowcount > 0:
                    new_chunks.append(chunk)
                else:
                    duplicate_count += 1

            except Exception as e:
                self.output.print('error', f"Error adding chunk: {e}")

        if new_chunks:
            self.chunks.extend(new_chunks)
            self.conn.commit()
            self._rebuild_search_indices()

            self.output.print('success', f"Added {len(new_chunks)} new chunks")
            if duplicate_count > 0:
                self.output.print('info', f"Skipped {duplicate_count} duplicate chunks")

    def _rebuild_search_indices(self):
        """Rebuild TF-IDF and semantic search indices"""
        if not self.chunks:
            return

        try:
            # Rebuild TF-IDF
            texts = [chunk.content for chunk in self.chunks]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            # Rebuild semantic embeddings if model is available
            if self.embedding_model is not None:
                self.output.print('processing', "Rebuilding semantic embeddings")
                self.embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

                # Build FAISS index
                dimension = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.embeddings)
                self.faiss_index.add(self.embeddings.astype('float32'))

        except Exception as e:
            self.output.print('error', f"Error rebuilding indices: {e}")
            self.output.print('warning', "Continuing with TF-IDF search only")

    def search_documents(self, query: str, method: str = 'hybrid', top_k: int = 5) -> List[Dict]:
        """Advanced document search with multiple methods"""
        if not self.chunks:
            return []

        results = []

        # Always use TF-IDF
        if method in ['tfidf', 'hybrid']:
            results.extend(self._search_tfidf(query, top_k))

        # Use semantic search only if embedding model is available
        if method in ['semantic', 'hybrid'] and self.embedding_model:
            semantic_results = self._search_semantic(query, top_k)
            for result in semantic_results:
                if not any(r['chunk_id'] == result['chunk_id'] and r['source'] == result['source'] for r in results):
                    results.append(result)
        elif method == 'semantic' and not self.embedding_model:
            self.output.print('warning', "Semantic search not available - using TF-IDF instead")
            results.extend(self._search_tfidf(query, top_k))

        # Sort by relevance score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return results[:top_k]

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        """Compatibility method for similarity search that returns document-like objects"""
        results = self.search_documents(query, 'hybrid', k)

        # Convert results to expected format
        output = []
        for result in results:
            # Create a simple object with page_content and metadata
            doc = type('Document', (), {
                'page_content': result['content'],
                'metadata': {
                    'source': result['source'],
                    'page': result.get('page_number', ''),
                    'file_type': result['file_type']
                }
            })()
            output.append(doc)

        return output

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        """Search with scores for compatibility"""
        results = self.search_documents(query, 'hybrid', k)

        # Convert results to expected format with scores
        output = []
        for result in results:
            doc = type('Document', (), {
                'page_content': result['content'],
                'metadata': {
                    'source': result['source'],
                    'page': result.get('page_number', ''),
                    'file_type': result['file_type']
                }
            })()
            output.append((doc, result['similarity_score']))

        return output

    def _search_tfidf(self, query: str, top_k: int) -> List[Dict]:
        """TF-IDF based search"""
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    chunk = self.chunks[idx]
                    results.append({
                        'content': chunk.content,
                        'source': chunk.source,
                        'chunk_id': chunk.chunk_id,
                        'file_type': chunk.file_type,
                        'page_number': chunk.page_number,
                        'similarity_score': float(similarities[idx]),
                        'search_method': 'tfidf'
                    })

            return results

        except Exception as e:
            self.output.print('error', f"TF-IDF search error: {e}")
            return []

    def _search_semantic(self, query: str, top_k: int) -> List[Dict]:
        """Semantic similarity search using embeddings"""
        if not self.embedding_model or self.faiss_index is None:
            self.output.print('warning', "Semantic search not available")
            return []

        try:
            query_embedding = self.embedding_model.encode([query])
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks) and score > 0.3:
                    chunk = self.chunks[idx]
                    results.append({
                        'content': chunk.content,
                        'source': chunk.source,
                        'chunk_id': chunk.chunk_id,
                        'file_type': chunk.file_type,
                        'page_number': chunk.page_number,
                        'similarity_score': float(score),
                        'search_method': 'semantic'
                    })

            return results

        except Exception as e:
            self.output.print('error', f"Semantic search error: {e}")
            return []

    def add_knowledge(self, category: str, key: str, value: str):
        """Add knowledge to the knowledge base"""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO knowledge (category, key, value)
                VALUES (?, ?, ?)
            ''', (category, key, value))

            self.knowledge_base[category][key] = value
            self.conn.commit()

        except Exception as e:
            self.output.print('error', f"Error adding knowledge: {e}")

    def get_knowledge(self, category: str = None) -> Dict:
        """Get knowledge from the knowledge base"""
        if category:
            return dict(self.knowledge_base.get(category, {}))
        return dict(self.knowledge_base)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {
            'total_chunks': len(self.chunks),
            'total_knowledge_items': sum(len(cat) for cat in self.knowledge_base.values()),
            'file_types': defaultdict(int),
            'sources': defaultdict(int),
            'knowledge_categories': list(self.knowledge_base.keys()),
            'search_capabilities': {
                'tfidf': True,
                'semantic': self.embedding_model is not None
            }
        }

        for chunk in self.chunks:
            stats['file_types'][chunk.file_type] += 1
            stats['sources'][chunk.source] += 1

        return stats


class EnhancedDocumentProcessor:
    """Advanced document processing with format detection"""

    def __init__(self, output_formatter: OutputFormatter):
        self.output = output_formatter
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.txt': self._process_text,
            '.md': self._process_text,
            '.py': self._process_text,
            '.js': self._process_text,
            '.html': self._process_text,
            '.css': self._process_text,
            '.json': self._process_text,
            '.xml': self._process_text,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.log': self._process_text
        }

    def process_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single file and return structured chunks"""
        try:
            suffix = file_path.suffix.lower()

            if suffix not in self.supported_formats:
                self.output.print('warning', f"Unsupported format: {suffix} for {file_path.name}")
                return []

            self.output.print('processing', f"Processing: {file_path.name}")

            chunks = self.supported_formats[suffix](file_path)

            # Add metadata to chunks
            for chunk in chunks:
                chunk.file_type = suffix[1:]
                chunk.timestamp = datetime.now().isoformat()
                chunk.hash_id = self._generate_hash(chunk.content, file_path.name, chunk.chunk_id)

            return chunks

        except Exception as e:
            self.output.print('error', f"Error processing {file_path.name}: {str(e)}")
            return []

    def _generate_hash(self, content: str, source: str, chunk_id: int) -> str:
        """Generate unique hash for chunk"""
        data = f"{content}{source}{chunk_id}".encode()
        return hashlib.md5(data).hexdigest()

    def _process_pdf(self, file_path: Path) -> List[DocumentChunk]:
        """Extract text from PDF"""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        page_chunks = self._chunk_text(text, file_path.name, page_num)
                        chunks.extend(page_chunks)
        except Exception as e:
            self.output.print('error', f"PDF processing error: {e}")
        return chunks

    def _process_docx(self, file_path: Path) -> List[DocumentChunk]:
        """Extract text from Word document"""
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = '\n'.join(paragraphs)
            return self._chunk_text(text, file_path.name)
        except Exception as e:
            self.output.print('error', f"DOCX processing error: {e}")
            return []

    def _process_text(self, file_path: Path) -> List[DocumentChunk]:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            return self._chunk_text(text, file_path.name)
        except Exception as e:
            self.output.print('error', f"Text processing error: {e}")
            return []

    def _process_csv(self, file_path: Path) -> List[DocumentChunk]:
        """Process CSV files"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            summary = f"CSV Summary:\nRows: {len(df)}, Columns: {len(df.columns)}\n"
            summary += f"Columns: {', '.join(df.columns)}\n\n"
            if len(df) > 0:
                summary += "Sample data:\n" + df.head().to_string()
            return self._chunk_text(summary, file_path.name)
        except Exception as e:
            self.output.print('error', f"CSV processing error: {e}")
            return []

    def _process_excel(self, file_path: Path) -> List[DocumentChunk]:
        """Process Excel files"""
        chunks = []
        try:
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_summary = f"Sheet: {sheet_name}\nRows: {len(df)}, Columns: {len(df.columns)}\n"
                if len(df) > 0:
                    sheet_summary += "Sample data:\n" + df.head().to_string()
                sheet_chunks = self._chunk_text(sheet_summary, f"{file_path.name}_{sheet_name}")
                chunks.extend(sheet_chunks)
        except Exception as e:
            self.output.print('error', f"Excel processing error: {e}")
        return chunks

    def _chunk_text(self, text: str, source: str, page_number: Optional[int] = None, chunk_size: int = 500) -> List[
        DocumentChunk]:
        """Split text into semantic chunks"""
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return []

        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence.split())

            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunk = DocumentChunk(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    file_type='',
                    page_number=page_number
                )
                chunks.append(chunk)

                current_chunk = [sentence]
                current_length = sentence_length
                chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunk = DocumentChunk(
                content=chunk_text,
                source=source,
                chunk_id=chunk_id,
                file_type='',
                page_number=page_number
            )
            chunks.append(chunk)

        return chunks


class ProfessionalRAGSystem:
    """Main RAG system with enhanced capabilities - FULLY LOCAL"""

    def __init__(self, adapter_path: str, output_formatter: OutputFormatter):
        self.adapter_path = adapter_path
        self.output = output_formatter
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Core components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.db = AdvancedRAGDatabase(output_formatter)
        self.doc_processor = EnhancedDocumentProcessor(output_formatter)

        # User facts
        self.facts = {}

    def initialize(self):
        """Initialize the complete system - LOCAL ONLY"""
        self.output.print('processing', "Initializing Professional RAG System (OFFLINE MODE)")

        # Initialize database and load existing data
        self.db.load_existing_data()

        # Initialize embeddings - LOCAL ONLY
        self.db.initialize_embeddings()

        # Load language model - LOCAL ONLY
        self._load_language_model()

        # AUTO-REBUILD INDICES if we have documents but no semantic search
        stats = self.db.get_statistics()
        if stats['total_chunks'] > 0 and not stats['search_capabilities'][
            'semantic'] and self.db.embedding_model is not None:
            self.output.print('warning', "Semantic search not available - rebuilding indices...")
            self.rebuild_indices()

        # Print system information
        self._print_system_banner()

    def _load_language_model(self):
        """Load the language model and adapter - FULLY LOCAL"""
        self.output.print('loading', "Loading local language model")

        # FORCE local model path - NO FALLBACK TO HUGGINGFACE
        model_path = "./hf_models/mistral-7b"
        if not os.path.exists(model_path):
            self.output.print('error', f"Local model not found at {model_path}")
            self.output.print('info', "Please download Mistral-7B model to ./hf_models/mistral-7b/")
            self.output.print('info', "Required files: config.json, pytorch_model*.bin, tokenizer files")
            raise FileNotFoundError(f"Local model required at: {model_path}")

        self.output.print('info', f"Base model: {model_path} (LOCAL)")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.output.print('info', f"GPU: {gpu_name}")
        else:
            self.output.print('info', "Using CPU (no GPU detected)")

        try:
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # Load base model - FORCE LOCAL FILES ONLY
            start_time = time.time()
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True  # CRITICAL: NO INTERNET ACCESS
            )

            load_time = time.time() - start_time
            self.output.print('success', f"Local model loaded in {load_time:.1f}s")

            # Load LoRA adapter
            self.output.print('processing', f"Loading local adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)

            # Load tokenizer - FORCE LOCAL FILES ONLY
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True  # CRITICAL: NO INTERNET ACCESS
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                return_full_text=False
            )

            adapter_name = os.path.basename(self.adapter_path)
            self.output.print('success', f"Loaded local adapter: {adapter_name}")

        except Exception as e:
            self.output.print('error', f"Local model loading error: {e}")
            self.output.print('info', "Ensure all model files are downloaded locally")
            raise

    def _print_system_banner(self):
        """Print comprehensive system banner"""
        stats = self.db.get_statistics()

        search_info = "TF-IDF"
        if stats['search_capabilities']['semantic']:
            search_info += " + Semantic"

        info = {
            'Adapter': os.path.basename(self.adapter_path),
            'Session': self.session_id,
            'Knowledge Base': f"{stats['total_chunks']} documents, {stats['total_knowledge_items']} facts",
            'Search Methods': search_info,
            'Mode': 'FULLY OFFLINE'
        }

        self.output.banner("Professional RAG System", info)

    def ingest_path(self, path: str):
        """Ingest documents from a file or directory - FIXED VERSION"""
        try:
            # Convert string path to Path object
            path_obj = Path(path)

            # Check if path exists
            if not path_obj.exists():
                self.output.print('error', f"Path not found: {path}")
                return

            # Handle file vs directory
            try:
                if path_obj.is_file():
                    self._ingest_file(path_obj)
                elif path_obj.is_directory():
                    self._ingest_directory(path_obj)
                else:
                    self.output.print('error', f"Invalid path type: {path}")
            except AttributeError as e:
                # Fallback for Path object issues
                self.output.print('error', f"Path object error: {str(e)}")
                # Try alternative approach
                if os.path.isfile(str(path_obj)):
                    self._ingest_file(path_obj)
                elif os.path.isdir(str(path_obj)):
                    self._ingest_directory(path_obj)
                else:
                    self.output.print('error', f"Cannot determine path type for: {path}")

        except Exception as e:
            self.output.print('error', f"Path handling error: {str(e)}")
            return

    def _ingest_file(self, file_path: Path):
        """Ingest a single file"""
        try:
            chunks = self.doc_processor.process_file(file_path)
            if chunks:
                self.db.add_chunks(chunks)
                self.output.print('success', f"Added: {file_path.name} ({len(chunks)} chunks)")
            else:
                self.output.print('warning', f"No content extracted from: {file_path.name}")
        except Exception as e:
            self.output.print('error', f"Error ingesting file {file_path.name}: {str(e)}")

    def _ingest_directory(self, dir_path: Path):
        """Ingest all supported files from a directory"""
        try:
            self.output.print('processing', f"Scanning directory: {dir_path}")

            supported_files = []

            # Use os.walk as fallback if Path.rglob fails
            try:
                for file_path in dir_path.rglob('*'):
                    if (file_path.is_file() and
                            file_path.suffix.lower() in self.doc_processor.supported_formats):
                        supported_files.append(file_path)
            except (AttributeError, OSError):
                # Fallback to os.walk
                for root, dirs, files in os.walk(str(dir_path)):
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.suffix.lower() in self.doc_processor.supported_formats:
                            supported_files.append(file_path)

            if not supported_files:
                self.output.print('warning', "No supported files found")
                return

            self.output.print('info', f"Found {len(supported_files)} files to process")

            total_chunks = 0
            successful_files = 0

            for file_path in supported_files:
                try:
                    chunks = self.doc_processor.process_file(file_path)
                    if chunks:
                        self.db.add_chunks(chunks)
                        total_chunks += len(chunks)
                        successful_files += 1
                        self.output.print('success', f"Added: {file_path.name} ({len(chunks)} chunks)")
                    else:
                        self.output.print('warning', f"No content from: {file_path.name}")
                except Exception as e:
                    self.output.print('error', f"Error processing {file_path.name}: {str(e)}")

            self.output.print('success', f"Processed {successful_files}/{len(supported_files)} files")
            self.output.print('info', f"Total chunks added: {total_chunks}")

        except Exception as e:
            self.output.print('error', f"Directory processing error: {str(e)}")

    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search the knowledge base"""
        return self.db.search_documents(query, 'hybrid', top_k)

    def debug_search(self, query: str, k: int = 10):
        """Debug search to see what's being retrieved"""
        try:
            results = self.db.similarity_search_with_score(query, k=k)

            print(f"\n{'=' * 60}")
            print(f"Query: '{query}'")
            print(f"Found {len(results)} results")
            print(f"{'=' * 60}\n")

            for i, (doc, score) in enumerate(results):
                print(f"Result {i + 1} (Score: {score:.3f}):")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Page: {doc.metadata.get('page', 'N/A')}")
                print(f"Content preview: {doc.page_content[:200]}...")
                print("-" * 40)

        except Exception as e:
            print(f"Debug search error: {e}")

    def query(self, question: str, k: int = 5):  # Reduced default k from 10 to 5
        """Query with enhanced context handling and dynamic token limits"""
        try:
            # Start timing
            start_time = time.time()

            # Get chunks for context - reduced default
            results = self.db.similarity_search(question, k=k)

            if not results:
                return "I couldn't find any relevant information."

            # Build rich context - OPTIMIZE: limit context size
            context_parts = []
            sources = set()
            total_context_length = 0
            max_context_length = 1500  # Reduced from 2000 for speed

            for doc in results:
                # Add the content if under limit
                if total_context_length + len(doc.page_content) < max_context_length:
                    context_parts.append(doc.page_content)
                    total_context_length += len(doc.page_content)

                # Track sources
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', '')
                sources.add(f"{source} (page {page})" if page else source)

            # Join all context
            full_context = "\n\n".join(context_parts[:3])  # Limit to first 3 chunks

            # Determine token limit based on query complexity
            max_tokens = self._determine_token_limit(question)

            # OPTIMIZE: Even simpler prompt
            prompt = f"""Context: {full_context}

Question: {question}
Answer:"""

            # Generate response with heavily optimized settings
            response = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.5,  # Lower temperature for faster, more focused generation
                do_sample=True,
                top_p=0.9,  # Tighter nucleus sampling
                top_k=40,  # Reduced from 50
                repetition_penalty=1.05,  # Further reduced
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                use_cache=True  # Enable KV cache
            )

            # Extract the answer text
            if isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], dict) and 'generated_text' in response[0]:
                    answer = response[0]['generated_text'].strip()
                else:
                    answer = str(response[0]).strip()
            else:
                answer = str(response).strip()

            # Clean up the answer
            answer = self._clean_response(answer)

            # Add sources if we gave a real answer
            if "don't have" not in answer.lower() and sources:
                answer += f"\n\nSources: {', '.join(sorted(list(sources)[:5]))}"  # Limit sources displayed

            # Log performance
            elapsed = time.time() - start_time
            if not self.output.quiet:
                self.output.print('info', f"Response generated in {elapsed:.1f}s ({max_tokens} max tokens)")

            return answer

        except Exception as e:
            self.output.print('error', f"Query error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def _determine_token_limit(self, question: str) -> int:
        """Determine appropriate token limit based on query complexity"""
        question_lower = question.lower()

        # Check for keywords indicating complexity - BALANCED FOR SPEED AND QUALITY
        if any(word in question_lower for word in ['detailed', 'comprehensive', 'thorough', 'complete', 'full']):
            return 768  # Increased from 512
        elif any(word in question_lower for word in ['summarize', 'summary', 'overview', 'explain']):
            return 512  # Increased from 384
        elif any(word in question_lower for word in ['list', 'enumerate', 'compare', 'contrast']):
            return 384  # Increased from 256
        elif any(word in question_lower for word in ['what is', 'who is', 'when', 'where', 'define']):
            return 256  # Increased from 128 to prevent cut-offs
        elif len(question.split()) > 20:  # Long questions might need detailed answers
            return 512  # Increased from 384
        else:
            return 384  # Increased default from 256

    def test_summary(self):
        """Test summarization with explicit context"""
        results = self.db.similarity_search("varieties religious experience", k=5)
        if results:
            context = "\n\n".join([doc.page_content for doc in results[:3]])

            test_prompt = f"""Based on this content from 'The Varieties of Religious Experience' by William James:

{context}

Please provide a summary of what this book is about."""

            response = self.pipeline(
                test_prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_full_text=False
            )

            print("\nTest Summary Result:")
            if isinstance(response, list) and len(response) > 0:
                print(response[0]['generated_text'])
            else:
                print(str(response))
        else:
            print("No results found for test summary")

    def learn_from_input(self, user_input: str):
        """Extract and learn facts from user input"""
        input_lower = user_input.lower()

        if "my name is" in input_lower:
            name = input_lower.split("my name is")[-1].strip().strip('.')
            if name:
                self.facts['user_name'] = name
                self.db.add_knowledge('personal', 'user_name', name)
                self.output.print('info', f"Learned: user_name = {name}")

        if "your name is" in input_lower:
            name = input_lower.split("your name is")[-1].strip().strip('.')
            if name:
                self.facts['assistant_name'] = name
                self.db.add_knowledge('personal', 'assistant_name', name)
                self.output.print('info', f"Learned: assistant_name = {name}")

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using enhanced prompt and constraints"""
        user_name = self.facts.get('user_name', 'User').title()
        assistant_name = self.facts.get('assistant_name', 'Margot')

        if not context:
            prompt = f"""You are {assistant_name}, a helpful AI assistant. Answer briefly and directly. Do not make up information or create fictional conversations.

{user_name}: {query}
{assistant_name}:"""
        else:
            context_text = ""
            for i, result in enumerate(context[:3], 1):
                content = result['content']
                if len(content) > 300:
                    content = content[:300] + "..."
                context_text += f"Source {i}: {content}\n\n"

            prompt = f"""You are {assistant_name}, a knowledgeable AI assistant. Use ONLY the provided context to answer the question. Follow these rules strictly:

CONTEXT:
{context_text.strip()}

RULES:
- Answer based ONLY on the context provided above
- Be concise and direct - give one clear answer
- If the context doesn't contain the answer, say "I don't have that information in my knowledge base"
- Do not make up information or create fictional conversations
- Do not continue past your answer
- Stop after answering the question

{user_name}: {query}
{assistant_name}:"""

        start_time = time.time()

        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=120,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                return_full_text=False
            )

            generation_time = time.time() - start_time
            answer = response[0]['generated_text'].strip()

            # Clean up response
            answer = self._clean_response(answer)

            # Print timing information
            if not self.output.quiet:
                if self.output.professional:
                    print(f"\nGeneration time: {generation_time:.2f}s")
                    print(f"Context sources: {len(context)} documents")
                elif not self.output.no_emojis:
                    print(f"\n⚡ {generation_time:.2f}s generation")

            return answer

        except Exception as e:
            self.output.print('error', f"Generation error: {e}")
            return "I apologize, but I encountered an error generating a response."

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # First, look for question patterns that indicate continuation
        question_patterns = [
            r'\n+(?:How|What|Why|When|Where|Who|Which|Could|Should|Would|Is|Are|Do|Does|Did|Can|Will)',
            r'.*\?$',  # Any question at the end - FIXED THIS LINE
            r'\n+[A-Z][^.!?]*\?',  # New line starting with capital and ending with ?
        ]

        # Check for these patterns and cut off before them
        for pattern in question_patterns:
            try:
                match = re.search(pattern, response)
                if match:
                    response = response[:match.start()].strip()
                    break
            except Exception:
                continue  # Skip pattern if regex fails

        # Also stop at common continuation patterns
        stop_patterns = [
            r'\nUser:.*',
            r'\nPatrick:.*',
            r'\nYou:.*',
            r'\nQuestion:.*',
            r'\nAnswer:.*',
            r'\nExplanation:.*',
            r'\nAssistant:.*',
            r'\nMargot:.*',
            r'\n\n[A-Z].*:',  # Any new speaker pattern
        ]

        for pattern in stop_patterns:
            response = re.split(pattern, response, 1)[0]

        # Clean up multiple line breaks
        response = re.sub(r'\n{3,}', '\n\n', response)

        # Remove trailing incomplete sentences
        lines = response.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            # Stop if we hit a meta-pattern
            if line and any(marker in line.lower() for marker in
                            ['user:', 'patrick:', 'you:', 'question:', 'human:', 'assistant:',
                             'margot:', 'answer:', 'explanation:']):
                break
            # Keep good lines
            if line:
                clean_lines.append(line)

        answer = '\n'.join(clean_lines).strip()

        # Final cleanup - remove any trailing incomplete sentence
        if answer and not answer[-1] in '.!?"\'':
            # Find the last complete sentence
            last_period = answer.rfind('.')
            last_exclaim = answer.rfind('!')
            last_question = answer.rfind('?')
            last_complete = max(last_period, last_exclaim, last_question)

            if last_complete > len(answer) * 0.7:  # Only trim if we're keeping most of the response
                answer = answer[:last_complete + 1].strip()

        if not answer:
            answer = "I understand your question, but I need more specific information to provide a helpful answer."

        return answer

    def show_stats(self):
        """Display system statistics"""
        stats = self.db.get_statistics()

        print()
        if self.output.professional:
            print("SYSTEM STATISTICS")
            print("-" * 40)
            print(f"Total Documents: {stats['total_chunks']}")
            print(f"Knowledge Items: {stats['total_knowledge_items']}")
            print(f"Session ID: {self.session_id}")
            print(f"Adapter: {self.adapter_path}")
            print(f"Search Capabilities: {list(k for k, v in stats['search_capabilities'].items() if v)}")
            print(f"Mode: FULLY OFFLINE")

            if stats['file_types']:
                print("\nFile Type Distribution:")
                for file_type, count in sorted(stats['file_types'].items()):
                    print(f"  {file_type}: {count}")
        else:
            emoji = "" if self.output.no_emojis else "📊 "
            print(f"{emoji}System Statistics:")
            print(f"  Documents: {stats['total_chunks']}")
            print(f"  Knowledge: {stats['total_knowledge_items']} items")
            print(f"  Adapter: {os.path.basename(self.adapter_path)}")
            print(f"  Mode: OFFLINE")

    def show_formats(self):
        """Show supported file formats"""
        formats = list(self.doc_processor.supported_formats.keys())

        if self.output.professional:
            print("\nSUPPORTED FILE FORMATS")
            print("-" * 30)
            for fmt in sorted(formats):
                print(f"  {fmt}")
        else:
            emoji = "" if self.output.no_emojis else "📄 "
            print(f"\n{emoji}Supported formats: {', '.join(sorted(formats))}")

    def query_stream(self, question: str, k: int = 5):
        """Query with streaming response generation"""
        try:
            # Start timing
            start_time = time.time()

            # Show thinking indicator
            if self.output.professional:
                print("\n[THINKING] Searching knowledge base...", end='', flush=True)
            else:
                emoji = "" if self.output.no_emojis else "🤔 "
                print(f"\n{emoji}Thinking...", end='', flush=True)

            # Get chunks for context
            results = self.db.similarity_search(question, k=k)

            # Clear thinking message
            print('\r' + ' ' * 50 + '\r', end='', flush=True)

            if not results:
                yield "I couldn't find any relevant information."
                return

            # Build context
            context_parts = []
            sources = set()
            total_context_length = 0
            max_context_length = 1500

            for doc in results:
                if total_context_length + len(doc.page_content) < max_context_length:
                    context_parts.append(doc.page_content)
                    total_context_length += len(doc.page_content)

                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', '')
                sources.add(f"{source} (page {page})" if page else source)

            full_context = "\n\n".join(context_parts[:3])

            # Determine token limit
            max_tokens = self._determine_token_limit(question)

            # Simple prompt
            prompt = f"""Context: {full_context}

Question: {question}
Answer:"""

            # Show generating indicator briefly
            if self.output.professional:
                print("[GENERATING] ", end='', flush=True)
            else:
                emoji = "" if self.output.no_emojis else "✨ "
                print(f"{emoji}Generating: ", end='', flush=True)

            # Small delay for visual effect
            time.sleep(0.3)

            # Configure streaming
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation kwargs
            generation_kwargs = dict(
                text_inputs=prompt,
                max_new_tokens=max_tokens,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

            # Start generation in a separate thread
            thread = Thread(target=self.pipeline, kwargs=generation_kwargs)
            thread.start()

            # Stream the output
            generated_text = ""
            first_chunk = True
            for text in streamer:
                # Clear the "Generating:" text on first real output
                if first_chunk and text.strip():
                    print('\r' + ' ' * 20 + '\r', end='', flush=True)
                    first_chunk = False

                generated_text += text
                # Clean as we stream to stop early if we detect a problem
                if any(marker in text.lower() for marker in ['how did', 'why is this', 'what is your', 'explain why']):
                    break
                yield text

            # Wait for generation to complete
            thread.join()

            # Add sources after streaming completes
            if sources and "don't have" not in generated_text.lower():
                yield f"\n\nSources: {', '.join(sorted(list(sources)[:5]))}"

            # Log performance
            elapsed = time.time() - start_time
            if not self.output.quiet:
                print()  # New line after streaming
                self.output.print('info', f"Response generated in {elapsed:.1f}s ({max_tokens} max tokens)")

        except Exception as e:
            yield f"\nError: {str(e)}"
            self.output.print('error', f"Streaming error: {e}")
            import traceback
            traceback.print_exc()

    def rebuild_indices(self):
        """Manually rebuild all search indices"""
        self.output.print('processing', "Rebuilding all search indices...")
        self.db._rebuild_search_indices()
        stats = self.db.get_statistics()
        self.output.print('success', f"Indices rebuilt for {stats['total_chunks']} documents")
        if stats['search_capabilities']['semantic']:
            self.output.print('success', "Semantic search is now available")
        else:
            self.output.print('warning', "Semantic search not available - check embedding model")

    def handle_command(self, command: str) -> Optional[str]:
        """Handle special commands"""
        parts = command.strip().split(maxsplit=1)
        if not parts:
            return None

        cmd = parts[0].lower()
        args = parts[1].split() if len(parts) > 1 else []

        if cmd == "debug":
            if not args:
                return "Usage: debug <query>"
            self.debug_search(" ".join(args))
            return None

        elif cmd == "test-summary":
            # Manually test summarization with explicit context
            self.test_summary()
            return None

        return None  # Not a special command

    def run_interactive_session(self):
        """Run the main interactive chat session"""
        self.output.print('info', "Ready for queries. Type 'help' for commands or 'quit' to exit.")

        while True:
            try:
                if self.output.professional:
                    user_input = input("\n> ").strip()
                else:
                    emoji = "" if self.output.no_emojis else "💬 "
                    user_input = input(f"\n{emoji}You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.output.print('info', "Goodbye!")
                    break

                elif user_input.lower() == 'help':
                    self._show_help()

                elif user_input.startswith('ingest '):
                    path = user_input[7:].strip().strip('"\'')
                    self.ingest_path(path)

                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    self._show_search_results(query)

                elif user_input.startswith('debug '):
                    query = user_input[6:].strip()
                    self.debug_search(query)

                elif user_input == 'test-summary':
                    self.test_summary()

                elif user_input == 'rebuild-indices':
                    self.rebuild_indices()

                elif user_input == 'stats':
                    self.show_stats()

                elif user_input == 'formats':
                    self.show_formats()

                else:
                    # Check if it's a special command
                    result = self.handle_command(user_input)
                    if result is not None:
                        print(result)
                    else:
                        # Regular query - STREAMING BY DEFAULT
                        self.learn_from_input(user_input)

                        # Check if user wants non-streaming (batch) mode
                        if user_input.startswith("batch "):
                            question = user_input[6:].strip()
                            response = self.query(question)

                            if self.output.professional:
                                print(f"\nResponse:\n{response}")
                            else:
                                assistant_name = self.facts.get('assistant_name', 'Margot')
                                emoji = "" if self.output.no_emojis else "🤖 "
                                print(f"\n{emoji}{assistant_name}: {response}")
                        else:
                            # DEFAULT: Stream the response
                            question = user_input

                            # Handle k= prefix for both streaming and batch
                            k_value = 5  # default
                            if question.startswith("k="):
                                parts = question.split(" ", 1)
                                if len(parts) >= 2:
                                    try:
                                        k_value = int(parts[0][2:])
                                        question = parts[1]
                                    except ValueError:
                                        pass

                            if self.output.professional:
                                print("\nResponse:")
                            else:
                                assistant_name = self.facts.get('assistant_name', 'Margot')
                                emoji = "" if self.output.no_emojis else "🤖 "
                                print(f"\n{emoji}{assistant_name}: ", end='', flush=True)

                            # Stream the response
                            for chunk in self.query_stream(question, k=k_value):
                                print(chunk, end='', flush=True)
                            print()  # Final newline

            except KeyboardInterrupt:
                self.output.print('info', "\nExiting...")
                break
            except Exception as e:
                self.output.print('error', f"Error: {str(e)}")

    def _show_help(self):
        """Display help information"""
        print()
        if self.output.professional:
            print("AVAILABLE COMMANDS")
            print("-" * 30)
            print("  ingest <path>           Ingest documents from file/directory")
            print("  search <query>          Search knowledge base")
            print("  debug <query>           Debug search results")
            print("  test-summary            Test summarization capability")
            print("  stats                   Show system statistics")
            print("  formats                 Show supported file formats")
            print("  help                    Show this help")
            print("  quit/exit               Exit system")
            print()
            print("For regular queries, just type your question.")
            print("System is running in FULLY OFFLINE mode.")
        else:
            emoji = "" if self.output.no_emojis else "📖 "
            print(f"{emoji}Available commands:")
            print("  ingest <path> - Add documents")
            print("  search <query> - Search knowledge")
            print("  debug <query> - Debug search")
            print("  test-summary - Test summarization")
            print("  stats - System statistics")
            print("  formats - File formats")
            print("  quit - Exit")
            print("  Mode: OFFLINE")

    def _show_search_results(self, query: str):
        """Display search results"""
        self.output.print('search', f"Searching for: {query}")

        results = self.search_knowledge_base(query, top_k=5)

        if not results:
            self.output.print('info', "No results found")
            return

        print()
        if self.output.professional:
            print(f"SEARCH RESULTS ({len(results)} found)")
            print("-" * 40)

            for i, result in enumerate(results, 1):
                print(f"\nResult {i}: {result['source']}")
                print(f"File type: {result['file_type']}")
                print(f"Relevance: {result['similarity_score']:.3f}")
                print(f"Method: {result['search_method']}")

                content = result['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"Content: {content}")
        else:
            emoji = "" if self.output.no_emojis else "🔍 "
            print(f"{emoji}Search Results:")

            for i, result in enumerate(results, 1):
                print(f"\n  {i}. {result['source']} ({result['file_type']}) - {result['similarity_score']:.2f}")
                content = result['content']
                if len(content) > 150:
                    content = content[:150] + "..."
                print(f"     {content}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Professional RAG System - FULLY OFFLINE - Advanced Document Analysis with Language Model Integration"
    )

    # Required arguments
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")

    # Output style arguments
    parser.add_argument("--professional", action="store_true", help="Use professional corporate output style")
    parser.add_argument("--no-emojis", action="store_true", help="Disable emoji output")
    parser.add_argument("--quiet", action="store_true", help="Minimize output messages")

    # Auto-ingestion arguments
    parser.add_argument("--ingest", nargs="+", metavar="PATH", help="Auto-ingest files/directories on startup")

    args = parser.parse_args()

    # Validate adapter path
    if not os.path.exists(args.adapter):
        print(f"Error: Adapter path not found: {args.adapter}")
        sys.exit(1)

    # Check for required local models
    model_path = "./hf_models/mistral-7b"
    embedding_path = "./sentence-transformer-model"

    if not os.path.exists(model_path):
        print(f"Error: Local Mistral model not found at {model_path}")
        print("Please download the Mistral-7B model files to ./hf_models/mistral-7b/")
        sys.exit(1)

    if not os.path.exists(embedding_path):
        print(f"Warning: Local embedding model not found at {embedding_path}")
        print("Semantic search will be unavailable. Only TF-IDF search will work.")
        print("To enable semantic search, place the sentence transformer model at ./sentence-transformer-model/")

    # Create output formatter
    output_formatter = OutputFormatter(
        professional=args.professional,
        no_emojis=args.no_emojis,
        quiet=args.quiet
    )

    # Initialize system
    try:
        output_formatter.print('info', "Starting FULLY OFFLINE RAG system")
        rag_system = ProfessionalRAGSystem(args.adapter, output_formatter)
        rag_system.initialize()

        # Auto-ingest if specified
        if args.ingest:
            for path in args.ingest:
                rag_system.ingest_path(path)

        # Run interactive session
        rag_system.run_interactive_session()

    except KeyboardInterrupt:
        output_formatter.print('info', "System shutdown requested")
    except Exception as e:
        output_formatter.print('error', f"System error: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if 'rag_system' in locals() and hasattr(rag_system.db, 'conn'):
            rag_system.db.conn.close()


if __name__ == "__main__":
    main()
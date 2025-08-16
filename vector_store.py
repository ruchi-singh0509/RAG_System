import os
import logging
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

from utils import (
    setup_environment, clean_text, log_processing_step,
    create_error_response, create_success_response
)

# Configure logging
logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store using FAISS for managing document embeddings and similarity search"""
    
    def __init__(self, persist_directory: str = None, model_name: str = None):
        """
        Initialize Vector Store with FAISS
        
        Args:
            persist_directory: Directory for FAISS index persistence
            model_name: Sentence transformer model name
        """
        # Setup environment
        setup_environment()
        
        # Configuration
        self.persist_directory = persist_directory or self._get_config('FAISS_PERSIST_DIRECTORY', './faiss_db')
        self.model_name = model_name or self._get_config('MODEL_NAME', 'all-MiniLM-L6-v2')
        
        # Create persist directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize FAISS index
        self._initialize_faiss_index()
        
        # Load metadata
        self._load_metadata()
        
        logger.info(f"FAISS Vector Store initialized with model: {self.model_name}")
    
    def _get_config(self, key: str, default_value: str) -> str:
        """Get configuration value from Streamlit secrets or environment variables"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                return st.secrets.get(key, default_value)
        except Exception:
            pass
        
        # Fallback to environment variables
        return os.getenv(key, default_value)
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded: {self.model_name} (dim: {self.embedding_dimension})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index"""
        try:
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            
            if os.path.exists(index_path):
                # Load existing index
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded existing FAISS index from {index_path}")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
                logger.info(f"Created new FAISS index with dimension {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _load_metadata(self):
        """Load document metadata"""
        try:
            metadata_path = os.path.join(self.persist_directory, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} documents")
            else:
                self.metadata = {}
                logger.info("No existing metadata found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """Save document metadata"""
        try:
            metadata_path = os.path.join(self.persist_directory, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata for {len(self.metadata)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _save_index(self):
        """Save FAISS index"""
        try:
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def add_document_chunks(self, chunks: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks with text and metadata
            document_id: Unique identifier for the document
            
        Returns:
            Dict with success status and metadata
        """
        try:
            log_processing_step(f"Adding {len(chunks)} chunks for document: {document_id}")
            
            if not chunks:
                return create_error_response("No chunks provided")
            
            # Prepare chunks for embedding
            texts = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '').strip()
                if not chunk_text:
                    continue
                
                # Clean text
                cleaned_text = clean_text(chunk_text)
                if not cleaned_text:
                    continue
                
                # Prepare metadata
                metadata = {
                    'document_id': document_id,
                    'chunk_index': i,
                    'chunk_type': chunk.get('type', 'text'),
                    'page_number': chunk.get('page_number', 0),
                    'source': chunk.get('source', 'unknown')
                }
                
                # Add any additional metadata from chunk
                for key, value in chunk.items():
                    if key not in ['text', 'type', 'page_number', 'source']:
                        metadata[key] = str(value)
                
                texts.append(cleaned_text)
                chunk_metadata.append({
                    'document_id': document_id,
                    'chunk_id': f"{document_id}_chunk_{i}",
                    'text': cleaned_text,
                    'metadata': metadata
                })
            
            if not texts:
                return create_error_response("No valid chunks to add")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            self.metadata[document_id] = {
                'chunks': chunk_metadata,
                'total_chunks': len(chunk_metadata),
                'added_at': str(np.datetime64('now'))
            }
            
            # Save index and metadata
            self._save_index()
            self._save_metadata()
            
            logger.info(f"Successfully added {len(chunk_metadata)} chunks for document: {document_id}")
            
            return create_success_response({
                'document_id': document_id,
                'chunks_added': len(chunk_metadata),
                'total_chunks': len(chunks)
            })
            
        except Exception as e:
            logger.error(f"Failed to add document chunks: {e}")
            return create_error_response(f"Failed to add document chunks: {str(e)}")
    
    def search_similar_chunks(self, query: str, top_k: int = 5, 
                            filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search for similar chunks based on query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters (not fully implemented for FAISS)
            
        Returns:
            Dict with search results
        """
        try:
            log_processing_step(f"Searching for similar chunks with query: {query[:100]}...")
            
            # Clean query
            cleaned_query = clean_text(query)
            if not cleaned_query:
                return create_error_response("Empty or invalid query")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([cleaned_query], show_progress_bar=False)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))
            
            # Get results
            formatted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Find the document and chunk for this index
                chunk_info = self._get_chunk_by_index(idx)
                if chunk_info:
                    # Apply metadata filter if provided
                    if filter_metadata:
                        chunk_metadata = chunk_info.get('metadata', {})
                        if not all(chunk_metadata.get(k) == v for k, v in filter_metadata.items()):
                            continue
                    
                    formatted_results.append({
                        'text': chunk_info['text'],
                        'metadata': chunk_info.get('metadata', {}),
                        'similarity_score': float(score),
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            
            return create_success_response({
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            })
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return create_error_response(f"Failed to search similar chunks: {str(e)}")
    
    def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict with document chunks
        """
        try:
            log_processing_step(f"Retrieving chunks for document: {document_id}")
            
            if document_id not in self.metadata:
                return create_error_response(f"Document {document_id} not found")
            
            doc_info = self.metadata[document_id]
            chunks = doc_info.get('chunks', [])
            
            # Format results
            formatted_chunks = []
            for chunk in chunks:
                formatted_chunks.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'chunk_id': chunk['chunk_id']
                })
            
            logger.info(f"Retrieved {len(formatted_chunks)} chunks for document: {document_id}")
            
            return create_success_response({
                'document_id': document_id,
                'chunks': formatted_chunks,
                'total_chunks': len(formatted_chunks)
            })
            
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return create_error_response(f"Failed to get document chunks: {str(e)}")
    
    def delete_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict with deletion status
        """
        try:
            log_processing_step(f"Deleting chunks for document: {document_id}")
            
            if document_id not in self.metadata:
                return create_error_response(f"Document {document_id} not found")
            
            # Get document chunks first to count them
            doc_info = self.metadata[document_id]
            chunk_count = doc_info.get('total_chunks', 0)
            
            # Remove from metadata
            del self.metadata[document_id]
            
            # Rebuild FAISS index without this document
            self._rebuild_index()
            
            logger.info(f"Successfully deleted {chunk_count} chunks for document: {document_id}")
            
            return create_success_response({
                'document_id': document_id,
                'chunks_deleted': chunk_count
            })
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            return create_error_response(f"Failed to delete document chunks: {str(e)}")
    
    def _rebuild_index(self):
        """Rebuild FAISS index from metadata"""
        try:
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            
            # Re-add all documents except deleted ones
            for doc_id, doc_info in self.metadata.items():
                chunks = doc_info.get('chunks', [])
                if chunks:
                    texts = [chunk['text'] for chunk in chunks]
                    embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
                    self.index.add(embeddings.astype('float32'))
            
            # Save updated index and metadata
            self._save_index()
            self._save_metadata()
            
            logger.info("FAISS index rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise
    
    def _get_chunk_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get chunk information by FAISS index"""
        try:
            current_index = 0
            for doc_id, doc_info in self.metadata.items():
                chunks = doc_info.get('chunks', [])
                for chunk in chunks:
                    if current_index == index:
                        return chunk
                    current_index += 1
            return None
        except Exception as e:
            logger.error(f"Failed to get chunk by index {index}: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Returns:
            Dict with collection statistics
        """
        try:
            stats = {
                'total_chunks': self.index.ntotal,
                'unique_documents': len(self.metadata),
                'collection_name': 'faiss_collection',
                'embedding_dimension': self.embedding_dimension,
                'embedding_model': self.model_name
            }
            
            logger.info(f"Collection stats: {stats}")
            
            return create_success_response(stats)
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return create_error_response(f"Failed to get collection stats: {str(e)}")
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all data from the collection
        
        Returns:
            Dict with clearing status
        """
        try:
            log_processing_step("Clearing entire collection")
            
            # Get count before clearing
            count = self.index.ntotal
            
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            
            # Clear metadata
            self.metadata = {}
            
            # Save empty index and metadata
            self._save_index()
            self._save_metadata()
            
            logger.info(f"Successfully cleared {count} chunks from collection")
            
            return create_success_response({
                'chunks_cleared': count,
                'collection_name': 'faiss_collection'
            })
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return create_error_response(f"Failed to clear collection: {str(e)}")
    
    def update_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update metadata for a specific chunk
        
        Args:
            chunk_id: Chunk identifier
            metadata: New metadata
            
        Returns:
            Dict with update status
        """
        try:
            log_processing_step(f"Updating metadata for chunk: {chunk_id}")
            
            # Find and update chunk metadata
            for doc_id, doc_info in self.metadata.items():
                chunks = doc_info.get('chunks', [])
                for chunk in chunks:
                    if chunk['chunk_id'] == chunk_id:
                        chunk['metadata'].update(metadata)
                        self._save_metadata()
                        
                        logger.info(f"Successfully updated metadata for chunk: {chunk_id}")
                        return create_success_response({
                            'chunk_id': chunk_id,
                            'metadata_updated': True
                        })
            
            return create_error_response(f"Chunk {chunk_id} not found")
            
        except Exception as e:
            logger.error(f"Failed to update chunk metadata: {e}")
            return create_error_response(f"Failed to update chunk metadata: {str(e)}")

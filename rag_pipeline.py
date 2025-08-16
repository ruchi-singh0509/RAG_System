
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import (
    setup_environment, clean_text, log_processing_step,
    create_error_response, create_success_response
)
from vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG Pipeline for processing queries and generating answers"""
    
    def __init__(self, vector_store: VectorStore, openai_api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_store: Vector store instance
            openai_api_key: OpenAI API key
            model_name: OpenAI model name
        """
        # Setup environment
        setup_environment()
        
        # Configuration
        self.vector_store = vector_store
        self.openai_api_key = openai_api_key or self._get_config('OPENAI_API_KEY', '')
        self.model_name = model_name
        
        # Initialize components
        self._initialize_llm()
        self._initialize_prompts()
        
        # Configuration
        self.top_k = int(self._get_config('TOP_K_RETRIEVAL', 5))
        self.max_tokens = int(self._get_config('MAX_TOKENS', 1000))
        self.temperature = float(self._get_config('TEMPERATURE', 0.7))
        
        logger.info(f"RAG Pipeline initialized with model: {self.model_name}")
    
    def _get_config(self, key: str, default_value: str = '') -> str:
        """Get configuration value from Streamlit secrets or environment variables"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                return st.secrets.get(key, default_value)
        except Exception:
            pass
        
        # Fallback to environment variables
        return os.getenv(key, default_value)
    
    def _initialize_llm(self):
        """Initialize language model"""
        try:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not provided")
            
            # Initialize chat model
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.openai_api_key
            )
            
            # Fallback to regular LLM if chat model fails
            try:
                self.llm.predict("Test")
            except Exception:
                self.llm = OpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=self.openai_api_key
                )
            
            logger.info(f"Language model initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise
    
    def _initialize_prompts(self):
        """Initialize prompt templates"""
        try:
            # Main QA prompt
            self.qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a helpful AI assistant that answers questions based on the provided context. 
Use only the information given in the context to answer the question. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
            )
            
            # Table analysis prompt
            self.table_prompt = PromptTemplate(
                input_variables=["table_data", "question"],
                template="""
You are an expert at analyzing table data. Please answer the question based on the table information provided.

Table Data:
{table_data}

Question: {question}

Answer:"""
            )
            
            # Summary prompt
            self.summary_prompt = PromptTemplate(
                input_variables=["content"],
                template="""
Please provide a concise summary of the following content:

{content}

Summary:"""
            )
            
            # Initialize chains
            self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
            self.table_chain = LLMChain(llm=self.llm, prompt=self.table_prompt)
            self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
            
            logger.info("Prompt templates and chains initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize prompts: {e}")
            raise
    
    def process_query(self, query: str, filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query and generate an answer
        
        Args:
            query: User query
            filter_metadata: Optional metadata filters for search
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            if not query.strip():
                return create_error_response("Empty query provided")
            
            # Clean query
            cleaned_query = clean_text(query)
            if not cleaned_query:
                return create_error_response("Query contains no valid text")
            
            # Log query processing
            log_processing_step("query_start", "query", {"query": cleaned_query})
            
            # Retrieve relevant chunks
            retrieval_result = self._retrieve_relevant_chunks(cleaned_query, filter_metadata)
            
            if not retrieval_result.get('success', False):
                return retrieval_result
            
            retrieved_chunks = retrieval_result.get('data', {}).get('results', [])
            
            if not retrieved_chunks:
                return create_error_response("No relevant information found for your query")
            
            # Generate answer
            answer_result = self._generate_answer(cleaned_query, retrieved_chunks)
            
            if not answer_result.get('success', False):
                return answer_result
            
            # Prepare response
            response_data = {
                'query': cleaned_query,
                'answer': answer_result.get('data', {}).get('answer', ''),
                'sources': retrieved_chunks,
                'total_sources': len(retrieved_chunks),
                'confidence_score': self._calculate_confidence_score(retrieved_chunks),
                'processing_time': answer_result.get('data', {}).get('processing_time', 0)
            }
            
            # Log query completion
            log_processing_step("query_complete", "query", {
                "query": cleaned_query,
                "sources_count": len(retrieved_chunks),
                "answer_length": len(response_data['answer'])
            })
            
            return create_success_response(response_data)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return create_error_response(f"Query processing failed: {str(e)}")
    
    def _retrieve_relevant_chunks(self, query: str, filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve relevant chunks from vector store"""
        try:
            # Search for similar chunks
            search_result = self.vector_store.search_similar_chunks(
                query=query,
                top_k=self.top_k,
                filter_metadata=filter_metadata
            )
            
            if not search_result.get('success', False):
                return search_result
            
            # Filter and rank results
            results = search_result.get('data', {}).get('results', [])
            filtered_results = self._filter_and_rank_results(results, query)
            
            return create_success_response({
                'results': filtered_results,
                'total_results': len(filtered_results),
                'query': query
            })
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return create_error_response(f"Retrieval failed: {str(e)}")
    
    def _filter_and_rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter and rank search results"""
        try:
            filtered_results = []
            
            for result in results:
                # Check similarity threshold
                similarity_score = result.get('similarity_score', 0)
                if similarity_score < 0.3:  # Minimum similarity threshold
                    continue
                
                # Check content quality
                content = result.get('content', '')
                if len(content.strip()) < 10:  # Minimum content length
                    continue
                
                # Add relevance score based on content type
                relevance_score = similarity_score
                content_type = result.get('metadata', {}).get('content_type', 'text')
                
                if content_type == 'table':
                    relevance_score *= 1.2  # Boost table content
                elif content_type == 'image':
                    relevance_score *= 0.8  # Reduce image content weight
                
                result['relevance_score'] = relevance_score
                filtered_results.append(result)
            
            # Sort by relevance score
            filtered_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            return results
    
    def _generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer from retrieved chunks"""
        try:
            import time
            start_time = time.time()
            
            # Prepare context from chunks
            context = self._prepare_context(chunks)
            
            # Determine answer type and generate response
            if self._is_table_query(query, chunks):
                answer = self._generate_table_answer(query, chunks)
            else:
                answer = self._generate_text_answer(query, context)
            
            processing_time = time.time() - start_time
            
            return create_success_response({
                'answer': answer,
                'processing_time': processing_time,
                'context_length': len(context)
            })
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return create_error_response(f"Answer generation failed: {str(e)}")
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks"""
        try:
            context_parts = []
            
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                content_type = metadata.get('content_type', 'text')
                
                # Format context based on content type
                if content_type == 'table':
                    context_part = f"Table {i+1}:\n{content}\n"
                elif content_type == 'image':
                    context_part = f"Image {i+1} (OCR text):\n{content}\n"
                else:
                    context_part = f"Text {i+1}:\n{content}\n"
                
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return ""
    
    def _is_table_query(self, query: str, chunks: List[Dict[str, Any]]) -> bool:
        """Check if query is related to table data"""
        table_keywords = ['table', 'data', 'row', 'column', 'cell', 'value', 'number', 'statistics']
        query_lower = query.lower()
        
        # Check query keywords
        if any(keyword in query_lower for keyword in table_keywords):
            return True
        
        # Check if chunks contain table data
        for chunk in chunks:
            content_type = chunk.get('metadata', {}).get('content_type', 'text')
            if content_type == 'table':
                return True
        
        return False
    
    def _generate_table_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate answer for table-related queries"""
        try:
            # Extract table data
            table_data = []
            for chunk in chunks:
                if chunk.get('metadata', {}).get('content_type') == 'table':
                    table_data.append(chunk.get('content', ''))
            
            if not table_data:
                return "I don't have enough table data to answer your question."
            
            # Combine table data
            combined_table_data = "\n\n".join(table_data)
            
            # Generate answer using table chain
            response = self.table_chain.run({
                'table_data': combined_table_data,
                'question': query
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating table answer: {e}")
            return "I encountered an error while analyzing the table data."
    
    def _generate_text_answer(self, query: str, context: str) -> str:
        """Generate answer for text-based queries"""
        try:
            # Generate answer using QA chain
            response = self.qa_chain.run({
                'context': context,
                'question': query
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating text answer: {e}")
            return "I encountered an error while generating the answer."
    
    def _calculate_confidence_score(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieved chunks"""
        try:
            if not chunks:
                return 0.0
            
            # Calculate average similarity score
            similarity_scores = [chunk.get('similarity_score', 0) for chunk in chunks]
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            
            # Calculate content quality score
            content_lengths = [len(chunk.get('content', '')) for chunk in chunks]
            avg_content_length = sum(content_lengths) / len(content_lengths)
            content_quality = min(avg_content_length / 500, 1.0)  # Normalize to 0-1
            
            # Calculate final confidence score
            confidence_score = (avg_similarity * 0.7) + (content_quality * 0.3)
            
            return min(confidence_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def generate_summary(self, document_id: str) -> Dict[str, Any]:
        """
        Generate summary for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary containing summary
        """
        try:
            # Get document chunks
            chunks_result = self.vector_store.get_document_chunks(document_id)
            
            if not chunks_result.get('success', False):
                return chunks_result
            
            chunks = chunks_result.get('data', {}).get('results', [])
            
            if not chunks:
                return create_error_response("No content found for the document")
            
            # Prepare content for summarization
            content_parts = []
            for chunk in chunks:
                content = chunk.get('content', '')
                if content:
                    content_parts.append(content)
            
            if not content_parts:
                return create_error_response("No valid content found for summarization")
            
            # Combine content
            combined_content = "\n\n".join(content_parts)
            
            # Limit content length for summarization
            if len(combined_content) > 4000:
                combined_content = combined_content[:4000] + "..."
            
            # Generate summary
            summary = self.summary_chain.run({'content': combined_content})
            
            return create_success_response({
                'document_id': document_id,
                'summary': summary.strip(),
                'content_length': len(combined_content),
                'chunks_count': len(chunks)
            })
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return create_error_response(f"Summary generation failed: {str(e)}")
    
    def search_documents(self, query: str, document_ids: List[str] = None) -> Dict[str, Any]:
        """
        Search across specific documents
        
        Args:
            query: Search query
            document_ids: List of document IDs to search in
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Prepare metadata filter
            filter_metadata = None
            if document_ids:
                filter_metadata = {'document_id': {'$in': document_ids}}
            
            # Process query
            return self.process_query(query, filter_metadata)
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return create_error_response(f"Document search failed: {str(e)}")
    
    def get_query_suggestions(self, partial_query: str) -> Dict[str, Any]:
        """
        Generate query suggestions based on partial query
        
        Args:
            partial_query: Partial user query
            
        Returns:
            Dictionary containing query suggestions
        """
        try:
            if not partial_query.strip():
                return create_success_response({'suggestions': []})
            
            # Get collection stats to understand available content
            stats_result = self.vector_store.get_collection_stats()
            
            if not stats_result.get('success', False):
                return create_error_response("Failed to get collection statistics")
            
            stats = stats_result.get('data', {})
            content_types = stats.get('content_types', {})
            
            # Generate suggestions based on content types
            suggestions = []
            
            if content_types.get('table', 0) > 0:
                suggestions.extend([
                    f"{partial_query} tables",
                    f"{partial_query} data",
                    f"{partial_query} statistics"
                ])
            
            if content_types.get('image', 0) > 0:
                suggestions.extend([
                    f"{partial_query} images",
                    f"{partial_query} charts",
                    f"{partial_query} diagrams"
                ])
            
            if content_types.get('text', 0) > 0:
                suggestions.extend([
                    f"{partial_query} information",
                    f"{partial_query} details",
                    f"{partial_query} content"
                ])
            
            # Remove duplicates and limit suggestions
            unique_suggestions = list(set(suggestions))[:5]
            
            return create_success_response({
                'suggestions': unique_suggestions,
                'partial_query': partial_query
            })
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return create_error_response(f"Failed to generate suggestions: {str(e)}")
    
    def evaluate_answer_quality(self, query: str, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the quality of generated answer
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_chunks: Retrieved chunks used for answer generation
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Calculate various quality metrics
            metrics = {}
            
            # Relevance score
            relevance_scores = [chunk.get('similarity_score', 0) for chunk in retrieved_chunks]
            metrics['avg_relevance'] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            
            # Answer length
            metrics['answer_length'] = len(answer)
            metrics['answer_length_score'] = min(len(answer) / 200, 1.0)  # Normalize to 0-1
            
            # Source diversity
            unique_docs = set(chunk.get('metadata', {}).get('document_id') for chunk in retrieved_chunks)
            metrics['source_diversity'] = len(unique_docs) / len(retrieved_chunks) if retrieved_chunks else 0
            
            # Content type coverage
            content_types = set(chunk.get('metadata', {}).get('content_type') for chunk in retrieved_chunks)
            metrics['content_type_coverage'] = len(content_types) / 3  # Normalize to 0-1 (text, table, image)
            
            # Overall quality score
            overall_score = (
                metrics['avg_relevance'] * 0.4 +
                metrics['answer_length_score'] * 0.2 +
                metrics['source_diversity'] * 0.2 +
                metrics['content_type_coverage'] * 0.2
            )
            
            metrics['overall_quality_score'] = overall_score
            
            return create_success_response({
                'query': query,
                'metrics': metrics,
                'chunks_count': len(retrieved_chunks)
            })
            
        except Exception as e:
            logger.error(f"Error evaluating answer quality: {e}")
            return create_error_response(f"Quality evaluation failed: {str(e)}")

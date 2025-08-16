
import streamlit as st
import os
import logging
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from utils import setup_environment, get_supported_formats
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Visual Document Analysis RAG System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

class RAGApp:
    def __init__(self):
        setup_environment()
        self._initialize_session_state()
        self._initialize_components()
    
    def _initialize_session_state(self):
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    def _initialize_components(self):
        try:
            # Initialize components step by step with better error handling
            st.info("🔄 Initializing system components...")
            
            # Step 1: Document Processor
            try:
                self.document_processor = DocumentProcessor()
                st.success("✅ Document Processor initialized")
            except Exception as e:
                st.error(f"❌ Document Processor failed: {str(e)}")
                raise
            
            # Step 2: Vector Store (most likely to fail)
            try:
                self.vector_store = VectorStore()
                st.success("✅ Vector Store initialized")
            except Exception as e:
                st.error(f"❌ Vector Store failed: {str(e)}")
                st.warning("💡 This is often due to PyTorch model loading issues")
                st.info("🔄 Trying alternative initialization...")
                
                # Try with a different model
                try:
                    import os
                    os.environ['MODEL_NAME'] = 'paraphrase-MiniLM-L3-v2'
                    self.vector_store = VectorStore()
                    st.success("✅ Vector Store initialized with fallback model")
                except Exception as fallback_e:
                    st.error(f"❌ Fallback initialization also failed: {str(fallback_e)}")
                    raise
            
            # Step 3: RAG Pipeline
            try:
                self.rag_pipeline = RAGPipeline(self.vector_store)
                st.success("✅ RAG Pipeline initialized")
            except Exception as e:
                st.error(f"❌ RAG Pipeline failed: {str(e)}")
                raise
            
            st.session_state.system_initialized = True
            st.success("🎉 All components initialized successfully!")
            
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            st.error("System initialization failed.")
    
    def run(self):
        st.markdown('<h1 class="main-header">📄 Visual Document Analysis RAG System</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        if st.session_state.system_initialized:
            tab1, tab2, tab3 = st.tabs(["📤 Upload", "🔍 Query", "📊 Analytics"])
            
            with tab1:
                self._render_upload_tab()
            with tab2:
                self._render_query_tab()
            with tab3:
                self._render_analytics_tab()
        else:
            st.error("System initialization failed.")
    
    def _render_sidebar(self):
        with st.sidebar:
            st.markdown("## 🎛️ System Status")
            
            if st.session_state.system_initialized:
                st.success("✅ System Ready")
                
                # Collection stats
                stats_result = self.vector_store.get_collection_stats()
                if stats_result.get('success', False):
                    stats = stats_result.get('data', {})
                    st.metric("Total Chunks", stats.get('total_chunks', 0))
                    st.metric("Documents", stats.get('unique_documents', 0))
            else:
                st.error("❌ System Error")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("🗑️ Clear Collection"):
                if st.session_state.system_initialized:
                    result = self.vector_store.clear_collection()
                    if result.get('success', False):
                        st.success("Collection cleared!")
                        st.rerun()
    
    def _render_upload_tab(self):
        st.markdown("## 📤 Upload Documents")
        
        supported_formats = get_supported_formats()
        all_formats = supported_formats['all']
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=[ext[1:] for ext in all_formats],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"**Selected {len(uploaded_files)} file(s)**")
            
            if st.button("🚀 Process Documents", type="primary"):
                self._process_uploaded_files(uploaded_files)
        
        # Show processed documents
        if st.session_state.processed_documents:
            st.markdown("### 📋 Processed Documents")
            for doc in st.session_state.processed_documents:
                with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Chunks", doc.get('chunks_count', 0))
                    with col2: st.metric("Tables", doc.get('tables_count', 0))
                    with col3: st.metric("Type", doc.get('document_type', 'Unknown').upper())
    
    def _process_uploaded_files(self, uploaded_files):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save file temporarily
                temp_path = f"./temp/{uploaded_file.name}"
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process document
                result = self.document_processor.process_document(temp_path)
                
                if result.get('success', False):
                    data = result.get('data', {})
                    chunks = data.get('chunks', [])
                    document_id = result.get('document_id')
                    
                    if chunks:
                        vector_result = self.vector_store.add_document_chunks(chunks, document_id)
                        
                        if vector_result.get('success', False):
                            doc_info = {
                                'document_id': document_id,
                                'filename': uploaded_file.name,
                                'file_size': uploaded_file.size,
                                'document_type': data.get('document_type', 'unknown'),
                                'chunks_count': len(chunks),
                                'tables_count': len(data.get('tables', [])),
                                'images_count': len(data.get('images', [])),
                                'processing_time': datetime.now().isoformat()
                            }
                            st.session_state.processed_documents.append(doc_info)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        status_text.text("Processing complete!")
        st.success(f"✅ Successfully processed documents!")
        st.rerun()
    
    def _render_query_tab(self):
        st.markdown("## 🔍 Query System")
        
        # Query input
        query = st.text_input("Enter your question:", placeholder="Ask about the uploaded documents...")
        top_k = st.selectbox("Top K Results", [3, 5, 10], index=1)
        
        if st.button("🔍 Search", type="primary") and query:
            self._process_query(query, top_k)
        
        # Query history
        if st.session_state.query_history:
            st.markdown("### 📜 Query History")
            for history_item in reversed(st.session_state.query_history[-5:]):
                with st.expander(f"Q: {history_item['query'][:50]}..."):
                    st.write(f"**Answer:** {history_item['answer']}")
                    st.write(f"**Confidence:** {history_item['confidence_score']:.2f}")
                    st.write(f"**Sources:** {history_item['total_sources']}")
    
    def _process_query(self, query: str, top_k: int):
        with st.spinner("Processing your query..."):
            result = self.rag_pipeline.process_query(query)
            
            if result.get('success', False):
                data = result.get('data', {})
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(data.get('answer', ''))
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Confidence", f"{data.get('confidence_score', 0):.2f}")
                with col2: st.metric("Sources", data.get('total_sources', 0))
                with col3: st.metric("Processing Time", f"{data.get('processing_time', 0):.2f}s")
                
                # Display sources
                sources = data.get('sources', [])
                if sources:
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(sources):
                        with st.expander(f"Source {i+1} (Score: {source.get('similarity_score', 0):.2f})"):
                            st.write(f"**Content:** {source.get('content', '')[:300]}...")
                
                # Add to history
                history_item = {
                    'query': query,
                    'answer': data.get('answer', ''),
                    'confidence_score': data.get('confidence_score', 0),
                    'total_sources': data.get('total_sources', 0),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.query_history.append(history_item)
                
            else:
                st.error(f"Query failed: {result.get('error_message', 'Unknown error')}")
    
    def _render_analytics_tab(self):
        st.markdown("## 📊 Analytics Dashboard")
        
        if not st.session_state.system_initialized:
            st.warning("System not initialized.")
            return
        
        # Get collection statistics
        stats_result = self.vector_store.get_collection_stats()
        
        if stats_result.get('success', False):
            stats = stats_result.get('data', {})
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Chunks", stats.get('total_chunks', 0))
            with col2: st.metric("Unique Documents", stats.get('unique_documents', 0))
            with col3: st.metric("Content Types", len(stats.get('content_types', {})))
            with col4: st.metric("Model", stats.get('embedding_model', 'Unknown'))
            
            # Content type distribution
            content_types = stats.get('content_types', {})
            if content_types:
                st.markdown("### 📈 Content Type Distribution")
                df = pd.DataFrame(list(content_types.items()), columns=['Type', 'Count'])
                st.bar_chart(df.set_index('Type'))
            
            # Document processing history
            if st.session_state.processed_documents:
                st.markdown("### 📋 Document Processing History")
                df = pd.DataFrame(st.session_state.processed_documents)
                st.dataframe(df[['filename', 'document_type', 'chunks_count', 'tables_count']], use_container_width=True)
            
            # Query analytics
            if st.session_state.query_history:
                st.markdown("### 🔍 Query Analytics")
                query_df = pd.DataFrame(st.session_state.query_history)
                st.line_chart(query_df.set_index('timestamp')['confidence_score'])
        
        else:
            st.error("Failed to retrieve analytics data")

def main():
    try:
        app = RAGApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()

# Visual Document Analysis RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that can process PDFs, images, and scanned documents to extract and retrieve information from tables, charts, and mixed text-image content.

## 🚀 Features

### Multi-format Document Processing
- **PDF Processing**: Extract text, tables, and images from PDF documents
- **Image Processing**: OCR for scanned documents and image-based content
- **Mixed Content**: Handle documents with text, tables, charts, and images

### Advanced Information Extraction
- **Table Extraction**: Automatically detect and extract data from tables
- **Chart Recognition**: Identify and analyze charts and graphs
- **OCR Integration**: High-accuracy text extraction from scanned documents
- **Visual Element Recognition**: Index and search visual elements

### Smart Retrieval System
- **Vector Database**: FAISS for efficient similarity search
- **Embedding Models**: Sentence Transformers for semantic understanding
- **Chunking Strategies**: Intelligent document chunking for optimal retrieval
- **Context-Aware Responses**: Generate relevant answers based on retrieved context

## 🛠️ Technical Architecture

### Core Components
1. **Document Processor**: Handles PDF, image, and scanned document ingestion
2. **OCR Engine**: Tesseract + EasyOCR for text extraction
3. **Table Extractor**: PDFPlumber + OpenCV for table detection
4. **Vector Database**: FAISS for storing and retrieving embeddings
5. **RAG Pipeline**: LangChain for retrieval-augmented generation
6. **Web Interface**: Streamlit for user-friendly interaction

### Key Technologies
- **Python 3.8+**
- **Streamlit**: Web application framework
- **LangChain**: RAG pipeline orchestration
- **FAISS**: Vector database
- **Sentence Transformers**: Embedding generation
- **OpenCV**: Image processing
- **Tesseract/EasyOCR**: OCR capabilities
- **PDFPlumber**: PDF text and table extraction

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd visual-document-rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```
## 🚀 Deployment

### Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Deploy!



### Using the System
1. **Upload Documents**: Upload PDFs, images, or scanned documents
2. **Process Documents**: The system automatically extracts text, tables, and visual elements
3. **Ask Questions**: Query the system about the uploaded documents
4. **Get Answers**: Receive context-aware responses with source citations

## 📊 Evaluation Metrics

The system evaluates performance using:
- **Retrieval Accuracy**: Precision and recall of relevant document chunks
- **Response Relevance**: Quality of generated answers
- **Processing Speed**: Document processing and query response times
- **OCR Accuracy**: Text extraction quality from scanned documents

## 🏗️ Project Structure

```
visual-document-rag/
├── app.py                 # Main Streamlit application
├── document_processor.py  # Document processing pipeline
├── ocr_engine.py         # OCR and text extraction
├── table_extractor.py    # Table detection and extraction
├── vector_store.py       # FAISS integration
├── rag_pipeline.py       # RAG query processing
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .env.example         # Environment variables template
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for text generation
- `FAISS_PERSIST_DIRECTORY`: Directory for FAISS persistence
- `MODEL_NAME`: Sentence transformer model name

### Model Configuration
- **Chunk Size**: 1000 characters with 200 character overlap
- **Top-k Retrieval**: 5 most relevant chunks

## 🎯 Use Cases

### Legal Domain
- Contract analysis and clause extraction
- Legal document search and retrieval
- Case law document processing

### Healthcare
- Medical report analysis
- Patient record processing
- Research paper information extraction

### Finance
- Financial report analysis
- Invoice and receipt processing
- Regulatory document compliance

### Education
- Research paper analysis
- Textbook content extraction
- Academic document processing







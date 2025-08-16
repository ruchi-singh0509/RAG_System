
import os
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and create necessary directories"""
    import streamlit as st
    
    # Try to load from Streamlit secrets first, then fall back to .env
    try:
        # Check if running in Streamlit
        if hasattr(st, 'secrets'):
            # Use Streamlit secrets
            faiss_dir = st.secrets.get('FAISS_PERSIST_DIRECTORY', './faiss_db')
        else:
            # Fall back to .env file for local development
            from dotenv import load_dotenv
            load_dotenv()
            faiss_dir = os.getenv('FAISS_PERSIST_DIRECTORY', './faiss_db')
    except Exception:
        # Fallback to .env file
        from dotenv import load_dotenv
        load_dotenv()
        faiss_dir = os.getenv('FAISS_PERSIST_DIRECTORY', './faiss_db')
    
    # Create necessary directories
    directories = [
        faiss_dir,
        './uploads',
        './temp',
        './logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup completed")

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings
            sentence_endings = ['.', '!', '?', '\n\n']
            for ending in sentence_endings:
                last_ending = text.rfind(ending, start, end)
                if last_ending > start + chunk_size // 2:  # Only break if it's not too early
                    end = last_ending + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def generate_document_id(file_path: str, content_hash: str = None) -> str:
    """Generate a unique document ID"""
    if content_hash:
        return f"{Path(file_path).stem}_{content_hash[:8]}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{Path(file_path).stem}_{timestamp}"

def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def validate_file_type(file_path: str, allowed_extensions: List[str]) -> bool:
    """Validate if file type is supported"""
    file_extension = Path(file_path).suffix.lower()
    return file_extension in allowed_extensions

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from file"""
    file_info = Path(file_path)
    
    metadata = {
        'filename': file_info.name,
        'file_extension': file_info.suffix.lower(),
        'file_size': get_file_size_mb(file_path),
        'file_size_formatted': format_file_size(os.path.getsize(file_path)),
        'created_time': datetime.fromtimestamp(file_info.stat().st_ctime).isoformat(),
        'modified_time': datetime.fromtimestamp(file_info.stat().st_mtime).isoformat(),
    }
    
    return metadata

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def create_temp_file(original_filename: str, content: bytes) -> str:
    """Create a temporary file with sanitized name"""
    temp_dir = Path('./temp')
    temp_dir.mkdir(exist_ok=True)
    
    sanitized_name = sanitize_filename(original_filename)
    temp_path = temp_dir / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_name}"
    
    with open(temp_path, 'wb') as f:
        f.write(content)
    
    return str(temp_path)

def cleanup_temp_files():
    """Clean up temporary files older than 1 hour"""
    temp_dir = Path('./temp')
    if not temp_dir.exists():
        return
    
    current_time = datetime.now()
    for temp_file in temp_dir.iterdir():
        if temp_file.is_file():
            file_age = current_time - datetime.fromtimestamp(temp_file.stat().st_mtime)
            if file_age.total_seconds() > 3600:  # 1 hour
                try:
                    temp_file.unlink()
                    logger.info(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

def log_processing_step(step: str, document_id: Optional[str] = None, details: Dict[str, Any] = None):
    """Log processing step for monitoring and debugging"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'step': step,
        'document_id': document_id,
        'details': details or {}
    }
    
    logger.info(f"Processing step: {step} for document {document_id}")
    
    # Save to log file
    log_file = Path('./logs/processing.log')
    log_file.parent.mkdir(exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{log_entry}\n")

def get_supported_formats() -> Dict[str, List[str]]:
    """Get supported file formats"""
    return {
        'documents': ['.pdf', '.txt', '.docx', '.doc'],
        'images': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
        'all': ['.pdf', '.txt', '.docx', '.doc', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    }

def is_image_file(file_path: str) -> bool:
    """Check if file is an image"""
    image_extensions = get_supported_formats()['images']
    return Path(file_path).suffix.lower() in image_extensions

def is_pdf_file(file_path: str) -> bool:
    """Check if file is a PDF"""
    return Path(file_path).suffix.lower() == '.pdf'

def create_error_response(error_message: str, error_type: str = "processing_error") -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        'success': False,
        'error_type': error_type,
        'error_message': error_message,
        'timestamp': datetime.now().isoformat()
    }

def create_success_response(data: Any, message: str = "Operation completed successfully") -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }

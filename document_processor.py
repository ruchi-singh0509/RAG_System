
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io

from utils import (
    setup_environment, clean_text, chunk_text, generate_document_id,
    calculate_content_hash, validate_file_type, extract_metadata,
    log_processing_step, get_supported_formats, is_image_file, is_pdf_file,
    create_error_response, create_success_response
)
from ocr_engine import OCREngine
from table_extractor import TableExtractor

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main document processor for handling multi-format documents"""
    
    def __init__(self, tesseract_cmd: str = None, languages: List[str] = None):
        """
        Initialize Document Processor
        
        Args:
            tesseract_cmd: Path to tesseract executable
            languages: List of languages for OCR
        """
        # Setup environment
        setup_environment()
        
        # Initialize components
        self.ocr_engine = OCREngine(tesseract_cmd, languages)
        self.table_extractor = TableExtractor(tesseract_cmd)
        
        # Configuration
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        
        logger.info("Document Processor initialized successfully")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document (PDF, image, or scanned document)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processed document data and metadata
        """
        try:
            # Validate file
            if not os.path.exists(file_path):
                return create_error_response(f"File not found: {file_path}")
            
            # Extract metadata
            metadata = extract_metadata(file_path)
            
            # Generate document ID
            document_id = generate_document_id(file_path)
            
            # Log processing start
            log_processing_step("start", document_id, {"file_path": file_path})
            
            # Process based on file type
            if is_pdf_file(file_path):
                result = self._process_pdf(file_path, document_id, metadata)
            elif is_image_file(file_path):
                result = self._process_image(file_path, document_id, metadata)
            else:
                return create_error_response(f"Unsupported file type: {metadata['file_extension']}")
            
            # Add common metadata
            result['document_id'] = document_id
            result['file_path'] = file_path
            result['metadata'] = metadata
            
            # Log processing completion
            log_processing_step("complete", document_id, {
                "chunks_count": len(result.get('chunks', [])),
                "tables_count": len(result.get('tables', [])),
                "text_length": len(result.get('full_text', ''))
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return create_error_response(f"Processing failed: {str(e)}")
    
    def _process_pdf(self, pdf_path: str, document_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF document"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text using multiple methods
            text_results = self._extract_text_from_pdf(pdf_path)
            
            # Extract tables
            tables = self.table_extractor.extract_tables_from_pdf(pdf_path)
            
            # Extract images
            images = self._extract_images_from_pdf(pdf_path)
            
            # Combine all text content
            full_text = self._combine_text_content(text_results)
            
            # Create chunks
            chunks = self._create_document_chunks(full_text, tables, images, document_id)
            
            # Calculate content hash
            content_hash = calculate_content_hash(full_text)
            
            return create_success_response({
                'full_text': full_text,
                'text_results': text_results,
                'tables': tables,
                'images': images,
                'chunks': chunks,
                'content_hash': content_hash,
                'document_type': 'pdf',
                'total_pages': len(text_results.get('pdfplumber_pages', [])),
                'tables_count': len(tables),
                'images_count': len(images)
            })
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return create_error_response(f"PDF processing failed: {str(e)}")
    
    def _process_image(self, image_path: str, document_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process image document"""
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Extract text using OCR
            ocr_result = self.ocr_engine.extract_text_from_image(image_path, method='hybrid')
            
            # Extract tables from image
            tables = self.table_extractor.extract_tables_from_image(image_path)
            
            # Get full text
            full_text = ocr_result.get('text', '')
            
            # Create chunks
            chunks = self._create_document_chunks(full_text, tables, [], document_id)
            
            # Calculate content hash
            content_hash = calculate_content_hash(full_text)
            
            return create_success_response({
                'full_text': full_text,
                'ocr_result': ocr_result,
                'tables': tables,
                'chunks': chunks,
                'content_hash': content_hash,
                'document_type': 'image',
                'ocr_confidence': ocr_result.get('confidence', 0),
                'tables_count': len(tables)
            })
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return create_error_response(f"Image processing failed: {str(e)}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF using multiple methods"""
        text_results = {}
        
        try:
            # Method 1: PDFPlumber
            pdfplumber_text = self._extract_text_pdfplumber(pdf_path)
            text_results['pdfplumber'] = pdfplumber_text
            
            # Method 2: PyMuPDF
            pymupdf_text = self._extract_text_pymupdf(pdf_path)
            text_results['pymupdf'] = pymupdf_text
            
            # Method 3: OCR for image-based PDFs
            ocr_text = self._extract_text_ocr_pdf(pdf_path)
            text_results['ocr'] = ocr_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        
        return text_results
    
    def _extract_text_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PDFPlumber"""
        try:
            text_content = []
            pages_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        pages_data.append({
                            'page_number': page_num + 1,
                            'text': page_text,
                            'bbox': page.bbox
                        })
            
            return {
                'full_text': '\n\n'.join(text_content),
                'pages': pages_data,
                'method': 'pdfplumber'
            }
            
        except Exception as e:
            logger.error(f"PDFPlumber text extraction failed: {e}")
            return {'full_text': '', 'pages': [], 'method': 'pdfplumber', 'error': str(e)}
    
    def _extract_text_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF"""
        try:
            text_content = []
            pages_data = []
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text:
                    text_content.append(page_text)
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'bbox': page.rect
                    })
            
            doc.close()
            
            return {
                'full_text': '\n\n'.join(text_content),
                'pages': pages_data,
                'method': 'pymupdf'
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF text extraction failed: {e}")
            return {'full_text': '', 'pages': [], 'method': 'pymupdf', 'error': str(e)}
    
    def _extract_text_ocr_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using OCR for image-based PDFs"""
        try:
            text_content = []
            pages_data = []
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # Higher resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Save temporary image
                temp_img_path = f"./temp/pdf_page_{page_num}.png"
                os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
                
                with open(temp_img_path, "wb") as f:
                    f.write(img_data)
                
                # Extract text using OCR
                ocr_result = self.ocr_engine.extract_text_from_image(temp_img_path)
                page_text = ocr_result.get('text', '')
                
                if page_text:
                    text_content.append(page_text)
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'ocr_confidence': ocr_result.get('confidence', 0),
                        'bbox': page.rect
                    })
                
                # Clean up temporary file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            
            doc.close()
            
            return {
                'full_text': '\n\n'.join(text_content),
                'pages': pages_data,
                'method': 'ocr'
            }
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return {'full_text': '', 'pages': [], 'method': 'ocr', 'error': str(e)}
    
    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Save image
                            img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                            img_path = f"./temp/{img_filename}"
                            os.makedirs(os.path.dirname(img_path), exist_ok=True)
                            
                            with open(img_path, "wb") as f:
                                f.write(img_data)
                            
                            # Extract text from image
                            ocr_result = self.ocr_engine.extract_text_from_image(img_path)
                            
                            images.append({
                                'page_number': page_num + 1,
                                'image_index': img_index + 1,
                                'image_path': img_path,
                                'image_size': len(img_data),
                                'ocr_text': ocr_result.get('text', ''),
                                'ocr_confidence': ocr_result.get('confidence', 0)
                            })
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF {pdf_path}: {e}")
        
        return images
    
    def _combine_text_content(self, text_results: Dict[str, Any]) -> str:
        """Combine text content from multiple extraction methods"""
        combined_text = []
        
        # Prioritize PDFPlumber text
        if text_results.get('pdfplumber', {}).get('full_text'):
            combined_text.append(text_results['pdfplumber']['full_text'])
        
        # Add PyMuPDF text if different
        pymupdf_text = text_results.get('pymupdf', {}).get('full_text', '')
        if pymupdf_text and pymupdf_text not in combined_text:
            combined_text.append(pymupdf_text)
        
        # Add OCR text for image-based content
        ocr_text = text_results.get('ocr', {}).get('full_text', '')
        if ocr_text and ocr_text not in combined_text:
            combined_text.append(ocr_text)
        
        return '\n\n'.join(combined_text)
    
    def _create_document_chunks(self, full_text: str, tables: List[Dict], images: List[Dict], document_id: str) -> List[Dict[str, Any]]:
        """Create document chunks for vector storage"""
        chunks = []
        chunk_id = 0
        
        # Create text chunks
        text_chunks = chunk_text(full_text, self.chunk_size, self.chunk_overlap)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'chunk_id': f"{document_id}_text_{i}",
                'content': chunk_text,
                'content_type': 'text',
                'chunk_index': chunk_id,
                'source': 'text_extraction',
                'metadata': {
                    'chunk_type': 'text',
                    'chunk_size': len(chunk_text),
                    'chunk_index': i
                }
            }
            chunks.append(chunk)
            chunk_id += 1
        
        # Create table chunks
        for i, table in enumerate(tables):
            if table.get('text'):
                table_chunk = {
                    'chunk_id': f"{document_id}_table_{i}",
                    'content': table['text'],
                    'content_type': 'table',
                    'chunk_index': chunk_id,
                    'source': 'table_extraction',
                    'metadata': {
                        'chunk_type': 'table',
                        'table_id': table.get('table_id', f'table_{i}'),
                        'rows': table.get('rows', 0),
                        'columns': table.get('columns', 0),
                        'method': table.get('method', 'unknown'),
                        'page_number': table.get('page_number', 0)
                    }
                }
                chunks.append(table_chunk)
                chunk_id += 1
        
        # Create image chunks
        for i, image in enumerate(images):
            if image.get('ocr_text'):
                image_chunk = {
                    'chunk_id': f"{document_id}_image_{i}",
                    'content': image['ocr_text'],
                    'content_type': 'image',
                    'chunk_index': chunk_id,
                    'source': 'image_extraction',
                    'metadata': {
                        'chunk_type': 'image',
                        'page_number': image.get('page_number', 0),
                        'image_index': image.get('image_index', 0),
                        'ocr_confidence': image.get('ocr_confidence', 0)
                    }
                }
                chunks.append(image_chunk)
                chunk_id += 1
        
        return chunks
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processing results
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(create_error_response(f"Processing failed: {str(e)}"))
        
        return results
    
    def get_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate processing summary
        
        Args:
            results: List of processing results
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_documents': len(results),
            'successful_processing': 0,
            'failed_processing': 0,
            'total_chunks': 0,
            'total_tables': 0,
            'total_images': 0,
            'total_text_length': 0,
            'document_types': {},
            'errors': []
        }
        
        for result in results:
            if result.get('success', False):
                summary['successful_processing'] += 1
                
                data = result.get('data', {})
                summary['total_chunks'] += len(data.get('chunks', []))
                summary['total_tables'] += len(data.get('tables', []))
                summary['total_images'] += len(data.get('images', []))
                summary['total_text_length'] += len(data.get('full_text', ''))
                
                doc_type = data.get('document_type', 'unknown')
                summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1
            else:
                summary['failed_processing'] += 1
                summary['errors'].append(result.get('error_message', 'Unknown error'))
        
        return summary
